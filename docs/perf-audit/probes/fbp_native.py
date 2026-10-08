"""Compare production FBP with direct native 2D BP and texture-based slice batching."""
import argparse
from contextlib import contextmanager
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXAMPLES / "tomography"))
import reconstruction


@contextmanager
def phase(name):
    from cupy.cuda import nvtx

    nvtx.RangePush(name)
    try:
        yield
    finally:
        nvtx.RangePop()


class NativeFBP:
    def __init__(self, theta, columns, library):
        self.library = library
        self.handle = library.fbp_create(columns, len(theta), theta.ctypes.data)
        if not self.handle:
            raise RuntimeError("native geometry initialization failed")

    def close(self):
        if self.handle:
            self.library.fbp_destroy(self.handle)
            self.handle = None

    def run(self, source, output, mode):
        import cupy as cp

        rows, angles, columns = source.shape
        size, response = reconstruction._fbp_filter(columns, self.filter_name, self.cutoff)
        response = cp.asarray(response)
        for block in reconstruction.slice_blocks(rows, self.filter_rows):
            spectrum = cp.fft.rfft(source[block], n=size, axis=-1)
            spectrum *= response
            batch = cp.fft.irfft(spectrum, n=size, axis=-1)
            result = self.library.fbp_run(self.handle, batch.data.ptr, output[block].data.ptr,
                block.stop - block.start, size, cp.cuda.get_current_stream().ptr, mode)
            if result:
                raise RuntimeError(f"native BP failed with code {result}")


def load_library():
    prefix = Path(sys.prefix)
    source = Path(__file__).with_suffix(".cu")
    output = source.with_name("_fbp_native.so")
    command = [str(prefix / "bin/nvcc"), "-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC",
               "-DASTRA_CUDA", "-I" + str(prefix / "include"), "-L" + str(prefix / "lib"),
               "-lastra", "-cudart=shared", "-gencode=arch=compute_75,code=sm_75",
               "-Xlinker=-rpath," + str(prefix / "lib"), str(source), "-o", str(output)]
    if not output.exists() or output.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(command, check=True)
    library = ctypes.CDLL(str(output))
    library.fbp_create.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    library.fbp_create.restype = ctypes.c_void_p
    library.fbp_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
    library.fbp_run.restype = ctypes.c_int
    library.fbp_destroy.argtypes = [ctypes.c_void_p]
    return library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--columns", type=int, default=512)
    parser.add_argument("--angles", type=int, default=360)
    parser.add_argument("--filters", nargs="+", default=["parzen"])
    parser.add_argument("--cutoff", type=float)
    parser.add_argument("--pattern", choices=["uniform", "signed", "edge-impulse"], default="uniform")
    parser.add_argument("--irregular-angles", action="store_true")
    parser.add_argument("--filter-rows", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.rows, args.columns, args.angles, args.filter_rows, args.repeats) < 1:
        parser.error("dimensions, filter batches and repeats must be positive")
    import astra
    import cupy as cp

    library = load_library()
    theta = np.linspace(0, np.pi, args.angles, endpoint=False, dtype=np.float32)
    if args.irregular_angles:
        theta += np.random.default_rng(92).uniform(-0.01, 0.01, args.angles).astype(np.float32)
    engine = NativeFBP(theta, args.columns, library)
    engine.filter_rows = args.filter_rows
    engine.cutoff = args.cutoff
    results = []
    try:
        with cp.cuda.Stream(non_blocking=True) as stream:
            for rows in args.rows:
                host = np.random.default_rng(71).random((rows, args.angles, args.columns), dtype=np.float32)
                if args.pattern == "signed":
                    host -= np.float32(0.5)
                elif args.pattern == "edge-impulse":
                    host.fill(0)
                    host[:, :, 0] = 1
                    host[:, ::2, -1] = -1
                source = cp.asarray(host)
                output = cp.empty((rows, args.columns, args.columns), dtype=cp.float32)
                for filter_name in args.filters:
                    engine.filter_name = filter_name
                    expected = reconstruction.reference_reconstruction(host, theta, 0,
                        reconstruction.configuration("fbp", filter_name, filter_cutoff=args.cutoff))
                    size, _ = reconstruction._fbp_filter(args.columns, filter_name, args.cutoff)
                    methods = {
                        "production": lambda: reconstruction.fbp_gpu(source, output, theta, 0, filter_name,
                            filter_cutoff=args.cutoff),
                        "native_2d": lambda: engine.run(source, output, 0),
                        "persistent_2d": lambda: engine.run(source, output, 2),
                        "batched_2d": lambda: engine.run(source, output, 1),
                    }
                    with patch.object(reconstruction, "FBP_FILTER_SAMPLES", args.filter_rows * args.angles * size):
                        for name, method in methods.items():
                            samples = []
                            for repeat in range(args.repeats + 1):
                                stream.synchronize()
                                with phase(f"{name}.rows{rows}.{filter_name}.repeat{repeat}"):
                                    start = time.perf_counter()
                                    method()
                                    stream.synchronize()
                                    elapsed = time.perf_counter() - start
                                if repeat:
                                    samples.append(elapsed)
                            actual = output.get()
                            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-6)
                            np.testing.assert_array_equal(source.get(), host)
                            result = dict(method=name, rows=rows, filter=filter_name,
                                seconds=samples, median_seconds=float(np.median(samples)),
                                reference_max_abs_error=float(np.max(np.abs(actual - expected))), validated=True)
                            print(json.dumps(result), flush=True)
                            results.append(result)
    finally:
        engine.close()
    args.output.write_text(json.dumps(dict(cupy=cp.__version__, astra=astra.__version__,
        shape=dict(columns=args.columns, angles=args.angles), filter_rows=args.filter_rows,
        pattern=args.pattern, irregular_angles=args.irregular_angles, filter_cutoff=args.cutoff, results=results,
        scope="GPU-resident filtering plus BP; persistent geometry creation, compilation, upload, "
              "and independent host ASTRA FBP_CUDA reference excluded. Native method keeps ASTRA "
              "BP kernels and per-slice stream/texture/constants. Batched method uses custom "
              "CUDA kernel with ASTRA 2D geometry and hardware texture interpolation."), indent=2) + "\n")


if __name__ == "__main__":
    main()
