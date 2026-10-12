"""Per-slice FBP and per-frame ingest costs at a target detector geometry, without transport.

Run it on each GPU the pipeline will use; NEXT.md says how to read the result.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction import FBP_FILTER_SAMPLES, _fbp_filter, fbp_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--columns", type=int, default=2048, help="detector width")
    parser.add_argument("--rows", type=int, default=2048, help="detector height, for the per-volume figures")
    parser.add_argument("--angles", type=int, default=2000)
    parser.add_argument("--fps", type=float, default=240, help="detector frame rate")
    parser.add_argument("--calibration-frames", type=int, default=100, help="darks plus flats per scan")
    parser.add_argument("--slices", type=int, default=4, help="slices reconstructed per sample")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import astra
    import cupy as cp

    cp.cuda.Device(args.gpu).use()
    stream = cp.cuda.Stream(non_blocking=True)

    def samples(operation, repeats=args.repeats):
        result = []
        for repeat in range(repeats + 1):
            stream.synchronize()
            start = time.perf_counter()
            operation()
            stream.synchronize()
            if repeat:  # the first call compiles kernels and plans FFTs
                result.append(time.perf_counter() - start)
        return result

    scan_seconds = (args.angles + args.calibration_frames) / args.fps
    reconstruction = []
    with stream:
        for binning in (1, 2, 4):
            columns, rows = args.columns // binning, args.rows // binning
            theta = np.linspace(0, np.pi, args.angles, endpoint=False, dtype=np.float32)
            sinogram = cp.random.random((args.slices, args.angles, columns), dtype=cp.float32)
            volume = cp.empty((args.slices, columns, columns), dtype=cp.float32)
            size, response = _fbp_filter(columns, "ram-lak", None)
            response = cp.asarray(response)
            block = max(1, FBP_FILTER_SAMPLES // (args.angles * size))

            def filtering():
                for start in range(0, args.slices, block):
                    spectrum = cp.fft.rfft(sinogram[start:start + block], n=size, axis=-1)
                    spectrum *= response
                    cp.fft.irfft(spectrum, n=size, axis=-1)[:, :, :columns].sum()

            total = samples(lambda: fbp_gpu(sinogram, volume, theta, args.gpu, "ram-lak"))
            filtered = samples(filtering)
            if not bool(cp.isfinite(volume).all()):
                raise AssertionError("FBP output must be finite")
            per_slice = float(np.median(total)) / args.slices
            reconstruction.append(dict(
                binning=binning, rows=rows, columns=columns, angles=args.angles,
                milliseconds_per_slice=per_slice * 1e3,
                filter_milliseconds_per_slice=float(np.median(filtered)) / args.slices * 1e3,
                seconds_per_volume=per_slice * rows,
                keeps_pace_with_back_to_back_scans=per_slice * rows < scan_seconds,
                samples_seconds=total))
            del sinogram, volume
            cp.get_default_memory_pool().free_all_blocks()

        frames = 16
        shape = (frames, args.rows, args.columns)
        raw = cp.random.randint(100, 60000, shape, dtype=cp.uint16)
        dark = cp.full(shape[1:], 90, cp.float32)
        flat = cp.full(shape[1:], 61000, cp.float32)
        corrected = cp.empty(shape, cp.float32)
        correct = cp.ElementwiseKernel(
            "uint16 raw, float32 dark, float32 flat", "float32 attenuation",
            "float t = (float(raw) - dark) / (flat - dark); "
            "attenuation = -logf(fminf(1.0f, fmaxf(1e-6f, t)));", "target_scale_correct")
        binned = cp.empty((frames, args.rows // 2, args.columns // 2), cp.uint32)

        def bin_frames():
            cp.sum(raw[:, :args.rows // 2 * 2, :args.columns // 2 * 2].reshape(
                frames, args.rows // 2, 2, args.columns // 2, 2), axis=(2, 4), dtype=cp.uint32, out=binned)

        rate = lambda seconds: frames / float(np.median(seconds))
        ingest = dict(correct_frames_per_second=rate(samples(lambda: correct(raw, dark, flat, corrected), 5)),
                      bin_2x2_frames_per_second=rate(samples(bin_frames, 5)))
        del corrected, binned
        cp.get_default_memory_pool().free_all_blocks()
        # A raw scan store in arrival order, as large as a quarter of the free memory allows.
        free = cp.cuda.Device(args.gpu).mem_info[0]
        stored = max(frames, min(args.angles, free // 4 // raw[0].nbytes))
        store = cp.zeros((stored, args.rows, args.columns), cp.uint16)
        slab_rows = min(64, args.rows)
        slab = cp.empty((slab_rows, stored, args.columns), cp.uint16)
        gather = samples(lambda: cp.copyto(slab, store[:, :slab_rows].transpose(1, 0, 2)), 5)
        append = samples(lambda: cp.copyto(store[:frames], raw), 5)
        ingest.update(slab_gather_gib_per_second=slab.nbytes / float(np.median(gather)) / 2**30,
                      batch_append_gib_per_second=raw.nbytes / float(np.median(append)) / 2**30,
                      stored_frames=int(stored))

    frame_bytes = args.rows * args.columns * 2
    device = cp.cuda.Device(args.gpu)
    result = dict(
        gpu=dict(index=args.gpu, memory_total_bytes=device.mem_info[1],
                 name=cp.cuda.runtime.getDeviceProperties(args.gpu)["name"].decode()),
        target=dict(rows=args.rows, columns=args.columns, angles=args.angles, frames_per_second=args.fps,
                    calibration_frames=args.calibration_frames, scan_seconds=scan_seconds,
                    detector_gib_per_second=frame_bytes * args.fps / 2**30,
                    raw_scan_gib=frame_bytes * (args.angles + args.calibration_frames) / 2**30,
                    corrected_float32_gib=2 * frame_bytes * args.angles / 2**30,
                    volume_float32_gib=4 * args.rows * args.columns**2 / 2**30),
        reconstruction=reconstruction, ingest=ingest,
        cupy=cp.__version__, astra=astra.__version__,
        scope="Existing fbp_gpu on random data; no transport, output or scientific validation")
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
