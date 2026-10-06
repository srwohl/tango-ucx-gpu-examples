"""Compare nvCOMP scalar calls with its list overload, using 16 independent buffers."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from processing_probe import fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("processing-batch-results.json"))
    args = parser.parse_args()
    if args.frames < 16 or args.repeats < 1:
        parser.error("frames must be at least16 and repeats positive")
    cuda_path = os.environ.get("CUDA_PATH")
    report = dict(status="ok", cuda_path=str(Path(cuda_path).resolve()) if cuda_path else None, rows=8, columns=64,
                  batch_size=16, input_bytes_per_frame=1024, cases=[], notes=[
                      "Both variants decode the same 16 independently aligned GPU buffers into 16 uint8 outputs.",
                      "No scratch copies, uploads, allocation, as_array construction or validation inside timing.",
                      "Both variants synchronize only once at end of each timed run.",
                      "The list API implementation is opaque; throughput does not establish GPU launch count."])
    try:
        import cupy as cp
        from nvidia import nvcomp
        report["nvcomp_version"] = nvcomp.__version__
        with tempfile.TemporaryDirectory(prefix="decode-batch-") as temp:
            directory = Path(temp) / "scan"
            directory.mkdir()
            meta, _, _, raw = fixture(directory, 8, 64, 16)
            frame = meta["frames"][2]
            packed = (directory / "compressed.bin").read_bytes()
            compressed = np.frombuffer(packed[frame["offset"]:frame["offset"] + frame["bytes"]], np.uint8).copy()
            report["compressed_bytes_per_frame"] = len(compressed)
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                inputs = [cp.asarray(compressed) for _ in range(16)]
                outputs = [cp.empty(raw.nbytes, dtype=cp.uint8) for _ in range(16)]
            codec = nvcomp.Codec(algorithm="LZ4", cuda_stream=stream.ptr,
                                 bitstream_kind=nvcomp.BitstreamKind.RAW, device_id=0)
            config = codec.decompression_config(codec.compression_config(raw.nbytes))
            batch_config = codec.decompression_config(codec.compression_config([raw.nbytes] * 16))
            sources = [nvcomp.as_array(array, cuda_stream=stream.ptr) for array in inputs]

            def scalar():
                for source, output in zip(sources, outputs):
                    codec.decode(source, out=output, decompression_config=config)

            def batch():
                codec.decode(sources, out=outputs, decompression_config=batch_config)

            count = args.frames // 16
            for name, operation in (("scalar_decode", scalar), ("list_decode16", batch)):
                samples = []
                for repeat in range(args.repeats + 1):
                    stream.synchronize()
                    start = time.perf_counter()
                    for _ in range(count):
                        operation()
                    stream.synchronize()
                    seconds = time.perf_counter() - start
                    sample = dict(seconds=seconds, frames=count * 16,
                                  microseconds_per_frame=seconds * 1e6 / (count * 16),
                                  input_MiB_per_second=raw.nbytes * count * 16 / seconds / 2**20)
                    if repeat == 0:
                        warmup = sample
                    else:
                        samples.append(sample)
                for output in outputs:
                    np.testing.assert_array_equal(output.get().view(np.uint16).reshape(raw.shape), raw)
                report["cases"].append(dict(stage=name, warmup=warmup, samples=samples,
                                             validated_output_buffers=len(outputs)))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
