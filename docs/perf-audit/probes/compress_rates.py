"""Scratch: nvCOMP GPU compression rate on 2048x2048 uint16 frames with Poisson noise."""
import time, json
import numpy as np, cupy as cp
from nvidia import nvcomp
print("nvcomp", nvcomp.__version__)
n = 16
yy, xx = cp.mgrid[0:2048, 0:2048].astype(cp.float32)
r = cp.sqrt((yy - 1024) ** 2 + (xx - 1024) ** 2)
trans = cp.where(r < 900, cp.exp(-2.0 * cp.sqrt(cp.maximum(0, 1 - (r / 900) ** 2))), 1.0)   # sphere
frames = cp.random.poisson(cp.broadcast_to(20000 * trans, (n, 2048, 2048))).astype(cp.uint16)
res = {}
def run(label, codec, arrays, raw_bytes, reps=4):
    rates, ratio = [], None
    for i in range(reps + 1):
        cp.cuda.get_current_stream().synchronize(); t = time.perf_counter()
        enc = codec.encode(arrays)
        size = sum(e.buffer_size for e in enc) if isinstance(enc, list) else enc.buffer_size
        cp.cuda.runtime.deviceSynchronize(); dt = time.perf_counter() - t
        if i: rates.append(round(raw_bytes / dt / 2**30, 2))
        ratio = round(raw_bytes / size, 2)
    res[label] = dict(GiB_per_s=rates, ratio=ratio)
raw_bytes = frames.nbytes
arrs = [nvcomp.as_array(frames[i].view(cp.uint8).reshape(-1)) for i in range(n)]
for algo in ("LZ4", "Bitcomp", "GDeflate", "Zstd"):
    for kind in ("NVCOMP_NATIVE", "RAW"):
        try:
            kw = dict(algorithm=algo, bitstream_kind=getattr(nvcomp.BitstreamKind, kind))
            if algo == "LZ4": kw["data_type"] = "<u2" if False else None
            kw = {k: v for k, v in kw.items() if v is not None}
            run(f"{algo}/{kind}", nvcomp.Codec(**kw), arrs, raw_bytes)
        except Exception as e:
            res[f"{algo}/{kind}"] = f"{type(e).__name__}: {str(e)[:90]}"
# byte-shuffled input (high bytes then low bytes): what a shuffle filter buys plain LZ4
sh = cp.ascontiguousarray(frames.view(cp.uint8).reshape(n, -1, 2).transpose(0, 2, 1)).reshape(n, -1)
try:
    run("LZ4/NATIVE byte-shuffled", nvcomp.Codec(algorithm="LZ4"), [nvcomp.as_array(sh[i]) for i in range(n)], raw_bytes)
except Exception as e:
    res["LZ4 shuffled"] = str(e)[:90]
for k, v in res.items(): print(k, v)
