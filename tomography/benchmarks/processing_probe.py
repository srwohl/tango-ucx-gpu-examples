"""Small GPU processing probe using application code, without transport/reconstruction."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan import _write_scan


def fixture(directory, rows, columns, frames):
    row, column = np.indices((rows, columns))
    dark = (16 + (row + column) % 7).astype(np.uint16)
    flat = dark + np.uint16(4000)
    raw = (dark + 1000 + (row * 31 + column * 17) % 2000).astype(np.uint16)
    theta = np.linspace(0, np.pi, frames, endpoint=False, dtype=np.float32)
    meta = _write_scan(directory, theta, dark, flat, (raw for _ in theta),
                       dict(type="processing-probe-synthetic"))
    return meta, dark, flat, raw


def measure(operation, stream, count, synchronized):
    stream.synchronize()
    start = time.perf_counter()
    for index in range(count):
        operation(index)
        if synchronized:
            stream.synchronize()
    stream.synchronize()
    elapsed = time.perf_counter() - start
    return dict(seconds=elapsed, frames=count, frames_per_second=count / elapsed,
                microseconds_per_frame=elapsed * 1e6 / count)


def run_gpu(meta, dark, flat, raw, directory, frames, repeats):
    import cupy as cp
    from processors import Processor

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        raw_gpu, dark_gpu, flat_gpu = (cp.asarray(a) for a in (raw, dark, flat))
        output = cp.empty(raw.shape, dtype=cp.float32)
        decoded = cp.empty(raw.shape, dtype=cp.uint16)
        batch_raw = cp.broadcast_to(raw_gpu, (16,) + raw.shape).copy()
        batch_output = cp.empty(batch_raw.shape, dtype=cp.float32)
        packed = (directory / "compressed.bin").read_bytes()
        compressed = [cp.asarray(np.frombuffer(packed[f["offset"]:f["offset"] + f["bytes"]],
                                               dtype=np.uint8).copy()) for f in meta["frames"]]
    correct = Processor("correct", meta, 0, stream.ptr, 1)
    decompress = Processor("decompress", meta, 0, stream.ptr, 1)

    def calibration():
        correct.begin_scan()
        for kind, source in enumerate((dark_gpu, flat_gpu)):
            correct.consume(source.data.ptr, source.nbytes, output.data.ptr, kind, 0, 0.)
        stream.synchronize()

    def correction(index):
        correct.consume(raw_gpu.data.ptr, raw_gpu.nbytes, output.data.ptr,
                        2, index, meta["theta"][index])

    def kernel(index):
        with stream:
            correct.correct(raw_gpu, correct.dark, correct.flat, output)

    def batch_kernel(index):
        with stream:
            correct.correct(batch_raw, correct.dark, correct.flat, batch_output)

    def decompression(index):
        f = meta["frames"][index]
        decompress.consume(compressed[index].data.ptr, f["bytes"], decoded.data.ptr,
                           f["kind"], f["projection"], f["theta"])

    results = []
    for stage, operation, count in (("correction_kernel", kernel, frames),
                                    ("correction_kernel_batch16", batch_kernel, frames // 16),
                                    ("correction_consume", correction, frames),
                                    ("decompression_consume", decompression, frames + 2)):
        for sync in (True, False):
            samples = []
            for repeat in range(repeats + 1):
                calibration()
                decompress.begin_scan()
                sample = measure(operation, stream, count, sync)
                if stage == "correction_kernel_batch16":
                    sample["kernel_calls"] = sample["frames"]
                    sample["frames"] *= 16
                    sample["frames_per_second"] *= 16
                    sample["microseconds_per_frame"] /= 16
                sample["input_MiB_per_second"] = raw.nbytes * sample["frames_per_second"] / 2**20
                if repeat == 0:
                    warmup = sample
                else:
                    samples.append(sample)
            results.append(dict(stage=stage, synchronization="per_frame" if sync else "end_only",
                                warmup=warmup, samples=samples))
    np.testing.assert_array_equal(decoded.get(), raw)
    expected = np.load(directory / "reference.npz")["sinogram"][:, -1, :]
    np.testing.assert_allclose(output.get(), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(batch_output.get(), np.broadcast_to(expected, batch_output.shape),
                               rtol=2e-6, atol=2e-6)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-check", action="store_true", help="Validate synthetic compressed input only; no timing")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 16 or args.repeats < 1:
        parser.error("frames must be at least 16 and repeats must be positive")
    report = dict(status="ok", cuda_path=os.environ.get("CUDA_PATH"), cases=[], notes=[
        "No transport, reconstruction, disk IO or initial host/device uploads in measured loops.",
        "Same stream/buffers reused safely; end_only queues frames, synchronizes once at end.",
        "correction_consume includes Python validation and borrowed-pointer views; kernel isolates launch/work.",
        "correction_kernel_batch16 uses one GPU kernel for 16 projections, rounds frame count down to multiple16.",
        "decompression_consume includes application scratch copy, nvCOMP decode and Python wrapper costs.",
        "Input MiB/s uses uncompressed uint16 bytes; correction outputs twice that byte count."])
    try:
        with tempfile.TemporaryDirectory(prefix="processing-probe-") as temp:
            for rows, columns in ((8, 64), (32, 128)):
                directory = Path(temp) / f"{rows}x{columns}"
                directory.mkdir()
                meta, dark, flat, raw = fixture(directory, rows, columns, args.frames)
                case = dict(rows=rows, columns=columns, input_bytes=raw.nbytes,
                            compressed_projection_bytes=meta["frames"][2]["bytes"])
                if args.cpu_check:
                    import lz4.block
                    packed = (directory / "compressed.bin").read_bytes()
                    f = meta["frames"][-1]
                    restored = lz4.block.decompress(packed[f["offset"]:f["offset"] + f["bytes"]],
                                                    uncompressed_size=raw.nbytes)
                    np.testing.assert_array_equal(np.frombuffer(restored, np.uint16).reshape(raw.shape), raw)
                    case["input_validation"] = "passed"
                else:
                    case["results"] = run_gpu(meta, dark, flat, raw, directory, args.frames, args.repeats)
                report["cases"].append(case)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
