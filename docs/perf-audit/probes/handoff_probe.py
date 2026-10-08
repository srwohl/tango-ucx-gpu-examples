"""Baseline probe: Processor.reconstruct on the host-buffered block path, NVTX-labelled.

Drives the unmodified production code on a non-blocking external stream, as C++ does.
"""
import json
import sys
import time

import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tomography"))
import astra
import cupy as cp
from cupy.cuda import nvtx

import host_buffering
import processors
from reconstruction import configuration

ROWS, ANGLES, COLUMNS, BLOCK = 128, 360, 512, 32


def labelled(name, function):
    def wrapper(*args, **kwargs):
        nvtx.RangePush(name)
        try:
            return function(*args, **kwargs)
        finally:
            nvtx.RangePop()
    return wrapper


astra.projector3d.direct_FP = labelled("astra_FP", astra.projector3d.direct_FP)
astra.projector3d.direct_BP = labelled("astra_BP", astra.projector3d.direct_BP)
astra.algorithm.run = labelled("astra_FBP_run", astra.algorithm.run)
host_buffering.BlockTransfers.prefetch = labelled("prefetch_h2d", host_buffering.BlockTransfers.prefetch)
processors.Processor._gpu_reconstruct_block = labelled(
    "reconstruct_block", processors.Processor._gpu_reconstruct_block)


def main():
    theta = np.linspace(0, np.pi, ANGLES, endpoint=False, dtype=np.float32)
    stream = cp.cuda.Stream(non_blocking=True)
    results = {}
    for method, options in (("sirt", configuration("sirt", iterations=20, slices_per_block=BLOCK)),
                            ("fbp", configuration("fbp", "parzen", slices_per_block=BLOCK))):
        scan = dict(rows=ROWS, angles=ANGLES, columns=COLUMNS, theta=theta.tolist(),
                    reconstruction=options,
                    buffering=dict(sinogram_memory="host", host_budget_bytes=1 << 30,
                                   pinned_budget_bytes=1 << 30))
        processor = processors.Processor("reconstruct", scan, 0, stream.ptr, 20)
        processor.sinogram[...] = np.random.default_rng(0).random(processor.sinogram.shape, np.float32)
        with stream:
            output = cp.empty((ROWS, COLUMNS, COLUMNS), dtype=cp.float32)
            samples = []
            for repeat in range(3):
                nvtx.RangePush(f"{method}_{'warmup' if repeat == 0 else 'measured'}")
                begin = time.perf_counter()
                processor.reconstruct(output)
                stream.synchronize()
                samples.append(time.perf_counter() - begin)
                nvtx.RangePop()
        processor.drain()
        results[method] = dict(warmup_s=samples[0], measured_s=samples[1:],
                               finite=bool(cp.isfinite(output).all()))
        del processor, output
    print(json.dumps(dict(shape=[ROWS, ANGLES, COLUMNS], block_rows=BLOCK, results=results), indent=2))


if __name__ == "__main__":
    main()
