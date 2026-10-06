"""Isolate existing host scan buffering and ASTRA FBP, without transport."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from host_buffering import HostScanBuffer
from reconstruction import fbp_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    import cupy as cp
    import astra

    scan = dict(rows=8, angles=96, columns=64)
    theta = np.linspace(0, np.pi, scan["angles"], endpoint=False, dtype=np.float32)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        sinogram = cp.ones((8, 96, 64), dtype=cp.float32)
        frame = cp.ones((8, 64), dtype=cp.float32)
        output = cp.empty((8, 64, 64), dtype=cp.float32)
    buffer = HostScanBuffer(scan, cp, stream)

    def append_scan():
        buffer.begin_scan()
        for projection in range(scan["angles"]):
            buffer.append(frame, projection)
        buffer.finish_downloads()

    results = []
    for label, operation in (("host_append", append_scan),
                             ("fbp", lambda: fbp_gpu(sinogram, output, theta, 0, "parzen"))):
        samples = []
        with stream:
            for repeat in range(args.repeats + 1):
                stream.synchronize()
                start = time.perf_counter()
                operation()
                stream.synchronize()
                elapsed = time.perf_counter() - start
                if repeat:
                    samples.append(elapsed)
                else:
                    warmup = elapsed
        if label == "host_append":
            np.testing.assert_array_equal(buffer.data, np.ones((8, 96, 64), np.float32))
            check = "host sinogram equals supplied ones frames"
        else:
            actual = output.get()
            if not np.isfinite(actual).all() or not np.any(actual != 0):
                raise AssertionError("FBP output must be finite and nonzero")
            check = "finite/nonzero output only; no scientific reference validation"
        mean = float(np.mean(samples))
        results.append(dict(stage=label, seconds_per_volume=mean,
                            microseconds_per_projection=mean * 1e6 / scan["angles"],
                            volumes_per_second=1 / mean, samples_seconds=samples,
                            discarded_warmup_seconds=warmup, correctness=check))
    result = dict(shape=scan, filter="parzen", repeats=args.repeats,
                  cupy=cp.__version__, astra=astra.__version__, results=results,
                  scope="Existing code only; no transport, block upload, or postprocessing")
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
