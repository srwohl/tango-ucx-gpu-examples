"""Scratch: per-slice FBP, correction and 2x2 binning cost at the target geometry (existing code)."""
import sys, time, json
import numpy as np
import cupy as cp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tomography"))
from reconstruction import fbp_gpu

def timed(fn, stream, repeats=3):
    out = []
    for r in range(repeats + 1):
        stream.synchronize(); t = time.perf_counter(); fn(); stream.synchronize()
        if r: out.append(time.perf_counter() - t)
    return out

res = {}
stream = cp.cuda.Stream(non_blocking=True)
with stream:
    for rows, angles, cols in ((4, 2000, 2048), (4, 2000, 1024), (4, 1000, 1024)):
        theta = np.linspace(0, np.pi, angles, endpoint=False, dtype=np.float32)
        sino = cp.random.random((rows, angles, cols), dtype=cp.float32)
        vol = cp.empty((rows, cols, cols), dtype=cp.float32)
        s = timed(lambda: fbp_gpu(sino, vol, theta, 0, "ram-lak"), stream)
        res[f"fbp_{angles}x{cols}"] = dict(ms_per_slice=[round(1e3 * x / rows, 1) for x in s],
                                           finite=bool(cp.isfinite(vol).all()))
        del sino, vol
        cp.get_default_memory_pool().free_all_blocks()
    # correction and binning on a batch of 16 raw 2048x2048 uint16 frames
    n = 16
    raw = cp.random.randint(100, 60000, (n, 2048, 2048), dtype=cp.uint16)
    dark = cp.full((2048, 2048), 90, cp.float32); flat = cp.full((2048, 2048), 61000, cp.float32)
    out = cp.empty((n, 2048, 2048), cp.float32)
    correct = cp.ElementwiseKernel("uint16 raw, float32 dark, float32 flat", "float32 a",
        "float t = (float(raw) - dark) / (flat - dark); a = -logf(fminf(1.0f, fmaxf(1e-6f, t)));", "probe_correct")
    s = timed(lambda: correct(raw, dark, flat, out), stream, 5)
    res["correct_fps_2048"] = [round(n / x) for x in s]
    binned = cp.empty((n, 1024, 1024), cp.uint32)
    def bin2():
        v = raw.reshape(n, 1024, 2, 1024, 2)
        cp.sum(v, axis=(2, 4), dtype=cp.uint32, out=binned)
    s = timed(bin2, stream, 5)
    res["bin2x2_fps_2048"] = [round(n / x) for x in s]
    # slab gather out of a projection-major raw store: 64 rows from every projection
    del out, binned
    cp.get_default_memory_pool().free_all_blocks()
    store = cp.zeros((400, 2048, 2048), cp.uint16)            # 3.1 GiB stand-in for a scan store
    slab = cp.empty((64, 400, 2048), cp.uint16)
    s = timed(lambda: cp.copyto(slab, store[:, 512:576, :].transpose(1, 0, 2)), stream, 5)
    res["slab_gather_GiB_per_s"] = [round(slab.nbytes / x / 2**30, 1) for x in s]
    s = timed(lambda: cp.copyto(store[:16], raw), stream, 5)
    res["ingest_copy_GiB_per_s"] = [round(raw.nbytes / x / 2**30, 1) for x in s]
print(json.dumps(res, indent=1))
