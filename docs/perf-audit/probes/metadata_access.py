"""Isolate Python record construction on real retained UCX batches, without payload work."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from network import environment


@contextmanager
def publisher():
    import tango

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryFile(mode="w+") as log:
        server = subprocess.Popen([
            str(ROOT / "build/opaque_publisher"), "local", "-nodb", "-dlist",
            "example/opaque/1", "-ORBendPoint", f"giop:tcp:127.0.0.1:{port}"],
            stdout=log, stderr=log)
        try:
            proxy = tango.DeviceProxy(f"127.0.0.1:{port}/example/opaque/1#dbase=no")
            proxy.set_timeout_millis(1000)
            deadline = time.monotonic() + 30
            while True:
                try:
                    proxy.ping()
                    break
                except tango.DevFailed:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("publisher readiness timeout")
                    time.sleep(.05)
            proxy.set_timeout_millis(30000)
            yield proxy
        except BaseException:
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def worker(args):
    if args.python_package is not None:
        sys.path.insert(0, str(args.python_package.resolve()))
    import tango_ucx

    results = []
    for requested in args.batch:
        with publisher() as proxy:
            proxy.command_inout("Configure", json.dumps(dict(source="host", lengths=[2048], budget=64 << 20)))
            begin = time.perf_counter()
            with tango_ucx.every(proxy, memory="host", budget="64MiB", batch=requested, max_wait=.01) as sub:
                connect_seconds = time.perf_counter() - begin
                proxy.command_inout("Start", requested)
                retained = sub.read(timeout=30)
                if retained is None:
                    raise RuntimeError(f"no batch: {sub.health()}")
                records = retained.records
                count = retained.frames
                if count != requested:
                    raise RuntimeError(f"expected one batch of {requested}, got {count}")
                np.testing.assert_array_equal(records["index"], np.arange(count))
                repeated = retained.records
                assert np.shares_memory(records, repeated) and not records.flags.writeable
                assert records.ctypes.data == repeated.ctypes.data
                assert int(records[-1]["seed"]) == ((count - 1) * 17 + 3) & 255

                def fresh_batch_once():
                    fresh = type(retained)(retained._native)
                    return sum(int(record["index"]) for record in fresh.records)

                def fresh_batch_per_frame():
                    fresh = type(retained)(retained._native)
                    return sum(int(fresh.records[index]["index"]) for index in range(count))

                variants = (
                    ("records_property_only", lambda: retained.records, 1),
                    ("native_records_property_only", lambda: retained._native.records, 1),
                    ("records_property_per_frame", lambda: sum(int(retained.records[index]["index"])
                                                             for index in range(count)), count),
                    ("records_once_per_batch", lambda: sum(int(record["index"])
                                                          for record in retained.records), count),
                    ("cached_records", lambda: sum(int(record["index"]) for record in records), count))
                variants += (("fresh_batch_records_once", fresh_batch_once, count),
                             ("fresh_batch_records_per_frame", fresh_batch_per_frame, count))
                if hasattr(retained._native, "index"):
                    variants += (("native_index", lambda: sum(retained._native.index(index)
                                                              for index in range(count)), count),)
                measurements = []
                for name, operation, units in variants:
                    operation()
                    samples = []
                    for repeat in range(3):
                        begin = time.perf_counter_ns()
                        for iteration in range(args.iterations):
                            operation()
                        samples.append((time.perf_counter_ns() - begin) / args.iterations)
                    measurements.append(dict(name=name, ns_per_operation=samples,
                                             median_ns_per_frame=float(np.median(samples)) / units))
                del records, repeated, retained, variants, operation
                terminal = sub.read(timeout=30)
                if terminal is not None or sub.outcome != "end":
                    raise RuntimeError(f"unexpected terminal state: {sub.health()}")
                health = sub.health()
                assert not health["failure"] and not health["quarantined_bytes"], health
                results.append(dict(frames=count, connect_seconds=connect_seconds,
                                    transport=health["transport"], zero_copy_records=True,
                                    measurements=measurements))
    import tango_ucx._core as native
    result = dict(results=results, native_module=native.__file__,
                  scope="Retained host UCX batch; metadata-only Python access. "
                  "Fresh-batch variants create a new public wrapper for the same native batch "
                  "each operation, including lazy records-view construction. "
                  "No steady-state transport throughput or C++ pipeline cost attribution. "
                  "Connect includes discovery and negotiation; first connection includes lazy imports.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 16, 128])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-package", type=Path,
                        help="package directory to prefer over the examples' staged installation")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if min(*args.batch, args.iterations) < 1 or max(args.batch) > 128:
        parser.error("batch must be 1..128 and iterations positive")
    if args.worker:
        worker(args)
    else:
        package_arguments = ([] if args.python_package is None else
                             ["--python-package", str(args.python_package.resolve())])
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--batch",
                        *map(str, args.batch), "--iterations", str(args.iterations),
                        "--output", str(args.output.resolve()), *package_arguments],
                       env=environment("auto"), check=True, timeout=120)


if __name__ == "__main__":
    main()
