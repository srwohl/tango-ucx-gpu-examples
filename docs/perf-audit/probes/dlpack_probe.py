"""What ASTRA 2.5 asks of DLPack, what it rejects, and what one exchange costs."""
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tomography"))
import astra
import cupy as cp

import host_buffering
import processors
from reconstruction import configuration


class Spy:
    """Forward the DLPack protocol to a CuPy array, recording what the consumer asked for."""
    calls = []

    def __init__(self, array):
        self.array = array

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()

    def __dlpack__(self, **kwargs):
        begin = time.perf_counter()
        capsule = self.array.__dlpack__(**kwargs)
        Spy.calls.append((kwargs, time.perf_counter() - begin))
        return capsule


def projector(rows, angles, columns):
    theta = np.linspace(0, np.pi, angles, endpoint=False, dtype=np.float32)
    return astra.projector3d.create(dict(
        type="cuda3d", option={"GPUindex": 0},
        ProjectionGeometry=astra.create_proj_geom("parallel3d", 1.0, 1.0, rows, columns, theta),
        VolumeGeometry=astra.create_vol_geom(columns, columns, rows)))


def attempt(label, function):
    try:
        function()
        print(f"  {label}: ACCEPTED")
    except Exception as error:
        print(f"  {label}: REJECTED ({type(error).__name__}: {str(error).splitlines()[0][:90]})")


stream = cp.cuda.Stream(non_blocking=True)
with stream:
    rows, angles, columns = 6, 5, 16
    p = projector(rows, angles, columns)
    volume = cp.zeros((rows, columns, columns), cp.float32)
    sinogram = cp.zeros((rows, angles, columns), cp.float32)

    print("1. What ASTRA passes to __dlpack__ (compute stream ptr = %#x):" % stream.ptr)
    astra.projector3d.direct_FP(p, Spy(volume), out=Spy(sinogram))
    for kwargs, seconds in Spy.calls:
        print(f"  kwargs={kwargs}  producer-side cost={seconds * 1e6:.0f} us")

    print("2. Axis meaning: light only volume z-slice 2, forward project")
    volume[2] = 1
    astra.projector3d.direct_FP(p, volume, out=sinogram)
    stream.synchronize()
    lit = cp.asnumpy(sinogram)
    print("  sinogram shape", lit.shape, "-> nonzero indices on axis 0:",
          sorted(set(np.nonzero(lit)[0].tolist())), "| nonzero angles on axis 1:",
          sorted(set(np.nonzero(lit)[1].tolist())))

    print("3. What ASTRA does with arrays that are not plain C-contiguous float32:")
    big = cp.zeros((rows, angles, 2 * columns), cp.float32)
    attempt("strided columns view", lambda: astra.projector3d.direct_FP(p, volume, out=big[:, :, ::2]))
    attempt("angles-first (angles, rows, columns)", lambda: astra.projector3d.direct_FP(
        p, volume, out=cp.zeros((angles, rows, columns), cp.float32)))
    attempt("Fortran order", lambda: astra.projector3d.direct_FP(
        p, volume, out=cp.zeros((rows, angles, columns), cp.float32, order="F")))
    attempt("float64", lambda: astra.projector3d.direct_FP(
        p, volume, out=cp.zeros((rows, angles, columns), cp.float64)))
    attempt("row-block slice [2:5] of both", lambda: astra.projector3d.direct_FP(
        projector(3, angles, columns), volume[2:5], out=sinogram[2:5]))

    print("4. Fixed cost of one direct call on a trivially small problem:")
    for name, call in (("direct_FP", lambda: astra.projector3d.direct_FP(p, volume, out=sinogram)),
                       ("direct_BP", lambda: astra.projector3d.direct_BP(p, sinogram, out=volume))):
        call()
        begin = time.perf_counter()
        for _ in range(50):
            call()
        print(f"  {name}: {(time.perf_counter() - begin) / 50 * 1e3:.2f} ms per call")
    flat, image = cp.zeros((angles, columns), cp.float32), cp.zeros((columns, columns), cp.float32)
    pg = astra.create_proj_geom("parallel", 1.0, columns, np.zeros(angles, np.float32))
    vg = astra.create_vol_geom(columns, columns)
    begin = time.perf_counter()
    for _ in range(50):
        a, b = astra.data2d.link("-sino", pg, flat), astra.data2d.link("-vol", vg, image)
        astra.data2d.delete([a, b])
    print(f"  data2d.link pair + delete: {(time.perf_counter() - begin) / 50 * 1e3:.2f} ms")

print("5. Layout of every array the production paths hand to ASTRA:")
seen = []
original = processors.Processor._gpu_reconstruct_block


def record(self, sinogram, volume):
    seen.append((sinogram, volume))
    return original(self, sinogram, volume)


processors.Processor._gpu_reconstruct_block = record
theta = np.linspace(0, np.pi, 40, endpoint=False, dtype=np.float32)
for memory, mode in (("gpu", "volume"), ("host", "volume"), ("gpu", "blocks"), ("host", "blocks")):
    options = configuration("sirt", iterations=1, slices_per_block=4)
    buffering = dict(sinogram_memory=memory, output_mode=mode, output_block_rows=4)
    scan = dict(rows=10, angles=40, columns=24, theta=theta.tolist(), reconstruction=options,
                buffering=buffering)
    processor = processors.Processor("reconstruct", scan, 0, stream.ptr, 1)
    processor.projections = 40
    seen.clear()
    with stream:
        if mode == "volume":
            output = cp.empty((10, 24, 24), cp.float32)
            processor.reconstruct(output)
        else:
            slot = cp.empty((4, 24, 24), cp.float32)  # stands in for the C++ publisher slot
            for start in range(0, 10, 4):
                processor.reconstruct_block(slot.data.ptr, start, min(4, 10 - start))
    processor.drain()
    summary = {(s.shape, s.flags.c_contiguous, s.dtype.name, s.strides == (s[0].nbytes, s[0, 0].nbytes, 4),
                v.shape, v.flags.c_contiguous, v.dtype.name) for s, v in seen}
    print(f"  sinogram_memory={memory} output_mode={mode}: {len(seen)} blocks")
    for s_shape, s_c, s_dtype, s_dense, v_shape, v_c, v_dtype in sorted(summary):
        print(f"    sinogram {s_shape} {s_dtype} C={s_c} dense_strides={s_dense} | volume {v_shape} {v_dtype} C={v_c}")
