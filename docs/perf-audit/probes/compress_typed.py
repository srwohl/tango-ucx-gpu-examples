import time
import cupy as cp
from nvidia import nvcomp
n = 16
yy, xx = cp.mgrid[0:2048, 0:2048].astype(cp.float32)
r = cp.sqrt((yy - 1024) ** 2 + (xx - 1024) ** 2)
trans = cp.where(r < 900, cp.exp(-2.0 * cp.sqrt(cp.maximum(0, 1 - (r / 900) ** 2))), 1.0)
frames = cp.random.poisson(cp.broadcast_to(20000 * trans, (n, 2048, 2048))).astype(cp.uint16)
for label, kw, arrs in (
    ("Bitcomp u16", dict(algorithm="Bitcomp", data_type="<u2"), [nvcomp.as_array(frames[i].reshape(-1)) for i in range(n)]),
    ("Bitcomp u16 algo1", dict(algorithm="Bitcomp", data_type="<u2", bitcomp_algo=1), [nvcomp.as_array(frames[i].reshape(-1)) for i in range(n)]),
    ("LZ4 u16", dict(algorithm="LZ4", data_type="<u2"), [nvcomp.as_array(frames[i].reshape(-1)) for i in range(n)]),
    ("Cascaded u16", dict(algorithm="Cascaded", data_type="<u2"), [nvcomp.as_array(frames[i].reshape(-1)) for i in range(n)]),
):
    try:
        codec = nvcomp.Codec(**kw); rates = []
        for i in range(4):
            cp.cuda.runtime.deviceSynchronize(); t = time.perf_counter()
            enc = codec.encode(arrs); size = sum(e.buffer_size for e in enc)
            cp.cuda.runtime.deviceSynchronize(); dt = time.perf_counter() - t
            if i: rates.append(round(frames.nbytes / dt / 2**30, 2))
        print(label, rates, "ratio", round(frames.nbytes / size, 2))
    except Exception as e:
        print(label, type(e).__name__, str(e)[:100])
