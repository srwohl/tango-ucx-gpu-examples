"""Measure production FBP scaling and label its existing library boundaries."""
import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
import functools
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tomography"))
import reconstruction


@contextmanager
def phase(name, counters):
    from cupy.cuda import nvtx

    nvtx.RangePush(name)
    begin = time.perf_counter_ns()
    try:
        yield
    finally:
        elapsed = time.perf_counter_ns() - begin
        nvtx.RangePop()
        counters[name]["calls"] += 1
        counters[name]["host_ns"] += elapsed


@contextmanager
def boundaries(counters):
    import astra
    import cupy as cp

    def wrap(name, function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            with phase(name, counters):
                return function(*args, **kwargs)
        return measured

    with ExitStack() as cleanup:
        for module, attribute, name in (
                (cp.fft, "rfft", "filter.rfft"),
                (cp.fft, "irfft", "filter.irfft"),
                (astra.data2d, "link", "astra.link"),
                (astra.algorithm, "create", "astra.create"),
                (astra.algorithm, "run", "astra.run"),
                (astra.algorithm, "delete", "astra.delete"),
                (cp.cuda.runtime, "deviceSynchronize", "handoff.device_sync")):
            cleanup.enter_context(patch.object(
                module, attribute, wrap(name, getattr(module, attribute))))
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--filter-rows", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--angles", type=int, default=360)
    parser.add_argument("--columns", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.rows, *args.filter_rows, args.angles, args.columns, args.repeats) < 1:
        parser.error("dimensions, filter batches and repeats must be positive")
    import astra
    import cupy as cp

    cp.cuda.Device(0).use()
    stream = cp.cuda.Stream(non_blocking=True)
    theta = np.linspace(0, np.pi, args.angles, endpoint=False, dtype=np.float32)
    size, _ = reconstruction._fbp_filter(args.columns, "parzen", None)
    results = []
    for rows in args.rows:
        host = np.random.default_rng(0).random((rows, args.angles, args.columns), dtype=np.float32)
        with stream:
            source = cp.asarray(host)
            output = cp.empty((rows, args.columns, args.columns), dtype=cp.float32)
        stream.synchronize()
        expected = reconstruction.reference_reconstruction(
            host, theta, 0, reconstruction.configuration("fbp", "parzen"))
        for filter_rows in args.filter_rows:
            samples = []
            counters = defaultdict(lambda: dict(calls=0, host_ns=0))
            with patch.object(reconstruction, "FBP_FILTER_SAMPLES", filter_rows * args.angles * size), stream:
                for repeat in range(args.repeats + 1):
                    with phase(f"baseline.rows{rows}.filter{filter_rows}", defaultdict(
                            lambda: dict(calls=0, host_ns=0))):
                        stream.synchronize()
                        begin = time.perf_counter()
                        reconstruction.fbp_gpu(source, output, theta, 0, "parzen")
                        stream.synchronize()
                        elapsed = time.perf_counter() - begin
                    if repeat:
                        samples.append(elapsed)
                with phase(f"instrumented.rows{rows}.filter{filter_rows}", counters), boundaries(counters):
                    reconstruction.fbp_gpu(source, output, theta, 0, "parzen")
                    stream.synchronize()
            actual = output.get()
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=2e-6)
            np.testing.assert_array_equal(source.get(), host)
            result = dict(rows=rows, filter_rows=filter_rows, seconds=samples,
                          median_seconds=float(np.median(samples)), boundaries=dict(counters),
                          reference_max_abs_error=float(np.max(np.abs(actual - expected))),
                          validated=True)
            results.append(result)
            print(json.dumps(result), flush=True)
        del source, output
    args.output.write_text(json.dumps(dict(
        shape=dict(angles=args.angles, columns=args.columns), cupy=cp.__version__,
        astra=astra.__version__, results=results,
        scope="GPU-resident production FBP only; uploads and reference verification outside timing. "
              "Baseline has outer NVTX labels only. Instrumented boundary host times include "
              "wrapper overhead and asynchronous submission, not isolated GPU execution; "
              "nested ranges must not be summed."), indent=2) + "\n")


if __name__ == "__main__":
    main()
