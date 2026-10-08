"""Decompress stage alone: microseconds per frame against the consume_many batch size."""
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
from processing_probe import fixture
from processors import Processor

ROWS, COLUMNS, FRAMES = 128, 512, 360
with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    meta, dark, flat, raw = fixture(directory, ROWS, COLUMNS, FRAMES)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        decoded = cp.empty((len(meta["frames"]),) + raw.shape, dtype=cp.uint16)
        packed = (directory / "compressed.bin").read_bytes()
        compressed = [cp.asarray(np.frombuffer(packed[f["offset"]:f["offset"] + f["bytes"]],
                                               dtype=np.uint8).copy()) for f in meta["frames"]]
    frames = [(source.data.ptr, f["bytes"], decoded[index].data.ptr, f["kind"], f["projection"], f["theta"])
              for index, (f, source) in enumerate(zip(meta["frames"], compressed))]
    decompress = Processor("decompress", meta, 0, stream.ptr, 1)
    results = {}
    for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256, 362):
        samples = []
        for repeat in range(4):
            decompress.begin_scan()
            stream.synchronize()
            begin = time.perf_counter()
            for start in range(0, len(frames), batch):
                decompress.consume_many(frames[start:start + batch])
            stream.synchronize()
            samples.append((time.perf_counter() - begin) / len(frames) * 1e6)
        np.testing.assert_array_equal(decoded[-1].get(), raw)
        results[batch] = [round(s) for s in samples[1:]]
        print(f"batch {batch:>3}: {results[batch]} us/frame  -> {1e6 / np.mean(samples[1:]):.0f} frames/s", flush=True)
    print(json.dumps(dict(frame=[ROWS, COLUMNS], compressed_bytes=meta["frames"][5]["bytes"])))
