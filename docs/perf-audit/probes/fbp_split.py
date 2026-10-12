"""Scratch: how the 69 ms per slice of fbp_gpu splits between filtering and ASTRA backprojection."""
import sys, time
import numpy as np, cupy as cp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tomography"))
from reconstruction import fbp_gpu, _fbp_filter, FBP_FILTER_SAMPLES
try:
    import tomocupy; print("tomocupy", tomocupy.__version__)
except Exception as e:
    print("tomocupy not importable:", type(e).__name__)
stream = cp.cuda.Stream(non_blocking=True)
with stream:
    for rows, angles, cols in ((4, 2000, 2048), (4, 2000, 1024)):
        theta = np.linspace(0, np.pi, angles, endpoint=False, dtype=np.float32)
        sino = cp.random.random((rows, angles, cols), dtype=cp.float32)
        vol = cp.empty((rows, cols, cols), dtype=cp.float32)
        size, response = _fbp_filter(cols, "ram-lak", None)
        response = cp.asarray(response)
        per = max(1, FBP_FILTER_SAMPLES // (angles * size))
        def filt():
            for s in range(0, rows, per):
                sp = cp.fft.rfft(sino[s:s + per], n=size, axis=-1); sp *= response
                b = cp.fft.irfft(sp, n=size, axis=-1)[:, :, :cols]
                b.sum()
        out = {}
        for name, fn in (("filter", filt), ("total", lambda: fbp_gpu(sino, vol, theta, 0, "ram-lak"))):
            ts = []
            for r in range(4):
                stream.synchronize(); t = time.perf_counter(); fn(); stream.synchronize()
                if r: ts.append(round(1e3 * (time.perf_counter() - t) / rows, 1))
            out[name] = ts
        print(angles, cols, "filter rows/block", per, out)
        del sino, vol; cp.get_default_memory_pool().free_all_blocks()
