"""Spike: TomocuPy's reconstruction classes on a borrowed stream and output slot, against ASTRA FBP.

Build the native modules first with ``probes/tomocupy/build.sh``. TomocuPy's file-based public
API and its global configuration are not used: only ``FBPFilter`` and the backprojection classes.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

PROBES = Path(__file__).resolve().parent
EXAMPLES = PROBES.parents[2]
sys.path[:0] = [str(EXAMPLES / "tomography"), str(PROBES), str(PROBES / "tomocupy/build")]
import reconstruction
from fbp_native import NativeFBP, load_library

TOMOCUPY_FILTERS = {"ram-lak": "ramp", "shepp-logan": "shepp", "hann": "hann", "parzen": "parzen"}
GUARD = 0x7FC0BEEF  # a quiet NaN pattern no reconstruction writes


@contextmanager
def phase(name):
    from cupy.cuda import nvtx

    nvtx.RangePush(name)
    try:
        yield
    finally:
        nvtx.RangePop()


class TomocupyFBP:
    """One fixed (chunk rows, angles, columns, theta) geometry, as TomocuPy's classes require.

    Follows ``BackprojFunctions.fbp_filter_center`` without ``args``/``params``, and filters in
    owned scratch: upstream writes the filtered data back into its input.
    """

    def __init__(self, method, rows, theta, columns, filter_name, center=None):
        import cupy as cp
        from tomocupy.reconstruction import fbp_filter, fourierrec, linerec, lprec

        angles = len(theta)
        self.rows, self.columns = rows, columns
        self.theta = cp.asarray(theta, dtype=cp.float32)
        self.size = 4 * columns
        self.pad = self.size // 2 - columns // 2
        center = columns / 2 if center is None else center
        if method == "lprec":
            center += 0.5  # as upstream: "consistence with the Fourier based method"
        self.filter = fbp_filter.FBPFilter(self.size, angles, rows, "float32")
        frequency = cp.fft.rfftfreq(self.size).astype(cp.float32)
        shift = cp.exp(-2 * cp.complex64(cp.pi * 1j) * (columns / 2 - center) * frequency)
        response = self.filter.calc_filter(TOMOCUPY_FILTERS[filter_name]) * shift
        self.response = cp.ascontiguousarray(cp.tile(response.astype(cp.complex64), (rows, 1)))
        self.padded = cp.empty((rows, angles, self.size), dtype=cp.float32)
        self.filtered = cp.empty((rows, angles, columns), dtype=cp.float32)
        if method == "fourierrec":
            self.backprojector = fourierrec.FourierRec(columns, angles, rows, self.theta, "float32")
        elif method == "linerec":
            self.backprojector = linerec.LineRec(self.theta, angles, angles, rows, rows, columns,
                                                 "float32")
        elif method == "lprec":
            self.backprojector = lprec.LpRec(columns, angles, rows, self.theta, "float32")
        else:
            raise ValueError(f"unknown TomocuPy method: {method}")

    def run(self, sinogram, output):
        import cupy as cp

        stream = cp.cuda.get_current_stream()
        pad, columns = self.pad, self.columns
        for block in reconstruction.slice_blocks(sinogram.shape[0], self.rows):
            data = sinogram[block]
            self.padded[:, :, pad:pad + columns] = data
            self.padded[:, :, :pad] = data[:, :, :1]
            self.padded[:, :, pad + columns:] = data[:, :, -1:]
            self.filter.filter(self.padded, self.response, stream)
            self.filtered[...] = self.padded[:, :, pad:pad + columns]
            self.backprojector.backprojection(output[block], self.filtered, stream)


def phantom(rows, angles, columns, theta):
    """Off-centre discs that differ per slice: analytic sinogram and the rasterised object."""
    discs = [(0.0, 0.0, 0.42, 1.0), (0.17, 0.09, 0.11, 0.8), (-0.21, 0.12, 0.07, -0.5),
             (0.05, -0.24, 0.05, 1.5), (-0.12, -0.15, 0.03, 2.0)]
    detector = (np.arange(columns, dtype=np.float32) - (columns - 1) / 2)[None, :]
    y, x = np.meshgrid(detector[0], detector[0], indexing="ij")
    cosine, sine = np.cos(theta)[:, None], np.sin(theta)[:, None]
    sinogram = np.zeros((rows, angles, columns), dtype=np.float32)
    truth = np.zeros((rows, columns, columns), dtype=np.float32)
    for row in range(rows):
        grow = 1 + 0.3 * row / max(1, rows - 1)
        for cx, cy, radius, density in discs:
            cx, cy, radius = cx * columns, cy * columns, radius * columns * (grow if cx else 1)
            offset = detector - (cx * cosine + cy * sine)
            sinogram[row] += (0.01 * density * 2) * np.sqrt(np.maximum(radius**2 - offset**2, 0))
            truth[row] += (0.01 * density) * ((x - cx)**2 + (y - cy)**2 < radius**2)
    return sinogram, truth


def aligned(candidate, reference, mask):
    """Best in-plane orientation, half-pixel grid offset and least-squares scale inside ``mask``."""
    def score(view):
        a, b = view[:, mask].astype(np.float64), reference[:, mask].astype(np.float64)
        scale = float((a * b).sum() / (a * a).sum())
        return float(np.linalg.norm(scale * a - b) / np.linalg.norm(b)), scale

    best = None
    for flip in (False, True):
        for turns in range(4):
            view = np.rot90(candidate, turns, axes=(1, 2))
            view = view[:, :, ::-1] if flip else view
            error, scale = score(view)
            if best is None or error < best[0]:
                best = (error, dict(turns=turns, flip=flip, shift=[0.0, 0.0], scale=scale), view)
    # TomocuPy centres its grid on pixel n/2 and ASTRA on (n - 1)/2: resample to compare.
    oriented = best[2]
    spectrum = np.fft.rfft2(oriented)
    ky = np.fft.fftfreq(oriented.shape[1])[:, None]
    kx = np.fft.rfftfreq(oriented.shape[2])[None, :]
    for dy in (-0.5, 0.0, 0.5):
        for dx in (-0.5, 0.0, 0.5):
            if dy or dx:
                view = np.fft.irfft2(spectrum * np.exp(-2j * np.pi * (ky * dy + kx * dx)),
                                     s=oriented.shape[1:]).astype(np.float32)
                error, scale = score(view)
                if error < best[0]:
                    best = (error, dict(best[1], shift=[dy, dx], scale=scale), view)
    return best[0], best[1], best[1]["scale"] * best[2]


class Borrowed:
    """A device allocation outside CuPy's pool with guard rows, as a publisher slot would be."""

    def __init__(self, shape, guard_rows=1):
        import cupy as cp
        from processors import array_at

        self.rows, self.guard = shape[0], guard_rows
        full = (shape[0] + 2 * guard_rows, *shape[1:])
        self.nbytes = int(np.prod(full)) * 4
        self.pointer = cp.cuda.runtime.malloc(self.nbytes)
        self.raw = array_at(self.pointer, full, cp.uint32, 0)
        self.raw.fill(GUARD)
        slot = self.pointer + guard_rows * int(np.prod(shape[1:])) * 4
        self.array = array_at(slot, shape, cp.float32, 0)

    def guards_intact(self):
        return bool((self.raw[:self.guard] == GUARD).all() and (self.raw[-self.guard:] == GUARD).all())

    def close(self):
        import cupy as cp

        del self.raw, self.array
        cp.cuda.runtime.free(self.pointer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--columns", type=int, default=512)
    parser.add_argument("--angles", type=int, default=360)
    parser.add_argument("--filter", default="parzen", choices=sorted(TOMOCUPY_FILTERS))
    parser.add_argument("--chunks", type=int, nargs="+", default=[32],
                        help="slices per TomocuPy call; its instances are fixed to one size")
    parser.add_argument("--methods", nargs="+", default=["fourierrec", "linerec", "lprec"])
    parser.add_argument("--baselines", nargs="*", default=["production", "persistent_2d"])
    parser.add_argument("--center-offset", type=float, default=-0.5,
                        help="TomocuPy rotation centre minus columns/2; -0.5 is the detector middle")
    parser.add_argument("--centers", type=float, nargs="*", default=[],
                        help="further centre offsets to compare, untimed")
    parser.add_argument("--negate-theta", action="store_true",
                        help="give TomocuPy -theta, which mirrors its rows")
    parser.add_argument("--filter-rows", type=int, default=32, help="ASTRA baselines' filter batch")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import astra
    import cupy as cp

    rows, angles, columns = args.rows, args.angles, args.columns
    theta = np.linspace(0, np.pi, angles, endpoint=False, dtype=np.float32)
    host, truth = phantom(rows, angles, columns, theta)
    tomocupy_theta = -theta if args.negate_theta else theta
    grid = np.arange(columns) - (columns - 1) / 2
    mask = (grid[:, None]**2 + grid[None, :]**2) < (0.47 * columns)**2
    corners = (grid[:, None]**2 + grid[None, :]**2) > (columns / 2 + 1)**2
    # Every slice is timed; a spread of them is compared, at their own indices.
    sample = np.unique(np.linspace(0, rows - 1, min(rows, 8)).astype(int))
    pool = cp.get_default_memory_pool()
    # The pipeline hands the processor a stream and output memory that C++ created.
    stream = cp.cuda.ExternalStream(cp.cuda.runtime.streamCreateWithFlags(1))
    results, volumes = [], {}

    def measure(name, method, borrowed, source, extra):
        samples, calls = [], []
        for repeat in range(args.repeats + 1):
            stream.synchronize()
            with phase(f"{name}.repeat{repeat}"):
                start = time.perf_counter()
                method()
                returned = time.perf_counter()
                stream.synchronize()
                elapsed = time.perf_counter() - start
            if repeat:
                samples.append(elapsed)
                calls.append(returned - start)
        volumes[name] = borrowed.array[sample].get()
        result = dict(method=name, rows=rows, seconds=samples,
                      median_seconds=float(np.median(samples)),
                      ms_per_slice=float(np.median(samples)) * 1e3 / rows,
                      median_call_return_seconds=float(np.median(calls)),
                      finite=bool(cp.isfinite(borrowed.array).all()),
                      input_unchanged=bool(np.array_equal(source.get(), host)),
                      output_guards_intact=borrowed.guards_intact(), **extra)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "seconds"}), flush=True)

    with stream:
        source = cp.asarray(host)
        borrowed = Borrowed((rows, columns, columns))
        # Label one kernel on each stream, so a trace can tell the borrowed stream's id.
        with phase("marker.borrowed-stream"):
            borrowed.raw[:1].fill(GUARD)
        with phase("marker.default-stream"), cp.cuda.Stream.null:
            cp.zeros(16, dtype=cp.float32).fill(1)
        size, _ = reconstruction._fbp_filter(columns, args.filter, None)
        with patch.object(reconstruction, "FBP_FILTER_SAMPLES", args.filter_rows * angles * size):
            if "production" in args.baselines:
                measure("production", lambda: reconstruction.fbp_gpu(
                    source, borrowed.array, theta, 0, args.filter), borrowed, source, {})
            if "persistent_2d" in args.baselines:
                engine = NativeFBP(theta, columns, load_library())
                engine.filter_rows, engine.cutoff, engine.filter_name = args.filter_rows, None, args.filter
                try:
                    measure("persistent_2d", lambda: engine.run(source, borrowed.array, 2),
                            borrowed, source, {})
                finally:
                    engine.close()
        for method in args.methods:
            for chunk in args.chunks:
                if rows % chunk:
                    parser.error("rows must be a multiple of every chunk")
                stream.synchronize()
                pool.free_all_blocks()
                free_before, pool_before = cp.cuda.runtime.memGetInfo()[0], pool.total_bytes()
                start = time.perf_counter()
                try:
                    engine = TomocupyFBP(method, chunk, tomocupy_theta, columns, args.filter,
                                         center=columns / 2 + args.center_offset)
                except (Exception, SystemExit) as error:
                    results.append(dict(method=f"tomocupy_{method}.chunk{chunk}", failed=repr(error)))
                    print(json.dumps(results[-1]), flush=True)
                    continue
                stream.synchronize()
                setup = time.perf_counter() - start
                engine.run(source, borrowed.array)
                stream.synchronize()
                device = free_before - cp.cuda.runtime.memGetInfo()[0]
                pooled = pool.total_bytes() - pool_before
                measure(f"tomocupy_{method}.chunk{chunk}", lambda: engine.run(source, borrowed.array),
                        borrowed, source, dict(chunk=chunk, setup_seconds=setup,
                        device_mib_per_slice=device / chunk / 2**20,
                        outside_cupy_pool_mib_per_slice=(device - pooled) / chunk / 2**20))
                del engine
        for offset in args.centers:
            for method in args.methods:
                engine = TomocupyFBP(method, args.chunks[0], tomocupy_theta, columns, args.filter,
                                     center=columns / 2 + offset)
                engine.run(source, borrowed.array)
                stream.synchronize()
                volumes[f"tomocupy_{method}.center{offset:+g}"] = borrowed.array[sample].get()
                del engine
        borrowed.close()

    # Orientation, scale and difference: against production FBP and against the object itself.
    reference = volumes.get("production")
    if reference is None:
        reference = reconstruction.reference_reconstruction(
            host[sample], theta, 0, reconstruction.configuration("fbp", args.filter))
    _, truth_orientation, truth = aligned(truth[sample], reference, mask)
    span = float(np.ptp(reference[:, mask]))
    accuracy = []
    for name, volume in volumes.items():
        error, orientation, view = aligned(volume, reference, mask)
        close = np.isclose(view[:, mask], reference[:, mask], rtol=3e-4, atol=2e-6)
        accuracy.append(dict(
            method=name, orientation_to_production=orientation,
            relative_l2_to_production=error,
            max_abs_to_production_over_range=float(np.abs(view - reference)[:, mask].max()) / span,
            within_pipeline_tolerance=float(close.mean()),
            relative_l2_to_object=float(np.linalg.norm((view - truth)[:, mask])
                                        / np.linalg.norm(truth[:, mask])),
            zero_fraction_in_corners=float((volume[:, corners] == 0).mean())))
        print(json.dumps(accuracy[-1]), flush=True)
    args.output.write_text(json.dumps(dict(
        cupy=cp.__version__, astra=astra.__version__, tomocupy_commit="d14ece0",
        shape=dict(rows=rows, angles=angles, columns=columns), filter=args.filter,
        filter_rows=args.filter_rows, compared_slices=sample.tolist(), tomocupy_center_offset=args.center_offset,
        tomocupy_negated_theta=args.negate_theta, results=results, accuracy=accuracy,
        truth_orientation_to_production=truth_orientation,
        scope="Timed: GPU-resident filtering, backprojection and completion on a caller-created "
              "non-blocking stream, into cudaMalloc memory outside the CuPy pool with guard rows. "
              "Excluded: upload, construction (reported as setup_seconds), compilation. "
              "device_mib_per_slice is the fall in free device memory from construction through "
              "one run; the part outside the CuPy pool is native cudaMalloc and cuFFT plans. "
              "Accuracy is inside 0.47 x columns of the axis, after the best in-plane orientation, "
              "half-pixel resampling and least-squares scale; the phantom is analytic discs."), indent=2) + "\n")


if __name__ == "__main__":
    main()
