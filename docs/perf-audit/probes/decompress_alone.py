"""Decompress stage alone at the pipeline's frame size: no transport, no other process, no poller."""
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[3] / "tomography"
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]
import cupy as cp
from cupy.cuda import nvtx
from processing_probe import fixture
from processors import Processor

ROWS, COLUMNS, FRAMES = 128, 512, 360
with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    meta, dark, flat, raw = fixture(directory, ROWS, COLUMNS, FRAMES)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        decoded = cp.empty(raw.shape, dtype=cp.uint16)
        packed = (directory / "compressed.bin").read_bytes()
        compressed = [cp.asarray(np.frombuffer(packed[f["offset"]:f["offset"] + f["bytes"]],
                                               dtype=np.uint8).copy()) for f in meta["frames"]]
    decompress = Processor("decompress", meta, 0, stream.ptr, 1)
    results = {}
    for mode in ("end_only", "per_frame"):
        samples = []
        for repeat in range(4):
            decompress.begin_scan()
            stream.synchronize()
            nvtx.RangePush(f"scan_{mode}_{'warmup' if repeat == 0 else 'measured'}")
            begin = time.perf_counter()
            for f, source in zip(meta["frames"], compressed):
                nvtx.RangePush("decompress.consume")
                decompress.consume(source.data.ptr, f["bytes"], decoded.data.ptr,
                                   f["kind"], f["projection"], f["theta"])
                nvtx.RangePop()
                if mode == "per_frame":
                    stream.synchronize()
            stream.synchronize()
            samples.append((time.perf_counter() - begin) / len(compressed) * 1e6)
            nvtx.RangePop()
        results[mode] = dict(warmup_us_per_frame=round(samples[0]),
                             measured_us_per_frame=[round(s) for s in samples[1:]])
    np.testing.assert_array_equal(decoded.get(), raw)
    print(json.dumps(dict(frame=[ROWS, COLUMNS], frames=len(compressed),
                          compressed_bytes=meta["frames"][5]["bytes"], results=results)))
