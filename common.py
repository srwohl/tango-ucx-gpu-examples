import argparse
from collections import deque
import json
import os
import time

import tango
import tango_ucx


def arguments(description, gpu=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("device", help="Tango device name or host:port/name#dbase=no")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--budget", default="32MiB")
    parser.add_argument("--delivery", choices=("every", "latest", "pull"), default="every")
    parser.add_argument("--range", type=int, default=16)
    parser.add_argument("--start", type=int, help="start this many frames AFTER this subscriber joins")
    parser.add_argument("--expected", type=int, help="require this many received frames")
    parser.add_argument("--timeout", type=float, default=60, help="seconds without a batch before failing")
    if gpu:
        parser.add_argument("--gpu", type=int, default=0, help="logical CUDA GPU number on this node")
        parser.add_argument("--inflight", type=int, default=4)
    return parser


def connect(args, memory, example="opaque-gpu-v1"):
    proxy = tango.DeviceProxy(args.device)
    proxy.set_timeout_millis(int(args.timeout * 1000))
    options = dict(memory=memory, budget=args.budget, batch=args.batch, max_wait=0.01,
                   allow_gpu_over_tcp=os.environ.get("EXAMPLES_NETWORK", "auto") != "rdma")
    if args.delivery == "pull":
        sub = tango_ucx.pull(proxy, args.range, **options)
    else:
        sub = getattr(tango_ucx, args.delivery)(proxy, **options)
    try:
        meta = json.loads(sub.description["application_text"])
        if meta.get("example") != example:
            raise ValueError(f"this subscriber expects the {example} publisher pattern")
        until = time.monotonic() + args.timeout
        while not sub.health()["transport"] and time.monotonic() < until:
            time.sleep(0.01)
        transport = sub.health()["transport"]
        policy = os.environ.get("EXAMPLES_NETWORK", "auto")
        names = {part.split("/", 1)[0] for part in transport.split(",")}
        if policy == "rdma" and ("tcp" in names or not any(n.startswith(("rc_", "dc_")) for n in names)):
            raise RuntimeError(f"RDMA was required; negotiated transports were {transport!r}")
        if policy == "tcp" and "tcp" not in names:
            raise RuntimeError(f"TCP was required; negotiated transports were {transport!r}")
        if not transport:
            raise RuntimeError("no negotiated transport reported")
        print(json.dumps({"ready": True, "memory": memory, "delivery": args.delivery,
                          "source": meta["source"], "transport": transport}), flush=True)
        if args.start is not None:
            proxy.command_inout("Start", args.start)
        return sub, meta
    except BaseException:
        sub.close()
        raise


def frames(sub, args):
    # A generator would retain its yielded batch while the caller uploads/analyzes it.
    # Returning from __next__ leaves ownership entirely with the example's caller.
    class Batches:
        def __iter__(self):
            return self

        def __next__(self):
            batch = sub.read(timeout=args.timeout)
            if batch is not None:
                return batch
            if sub.outcome is not None:
                if sub.outcome != "end":
                    raise RuntimeError(f"{sub.outcome}: {sub.health()}")
                raise StopIteration
            raise TimeoutError(f"no batch in {args.timeout}s: {sub.health()}")

    return Batches()


def records(batch, meta):
    result = []
    for row in batch.records:
        index, seed, size = int(row["index"]), int(row["seed"]), int(row["bytes"])
        if seed != (index * 17 + 3) & 255 or size != meta["lengths"][index % len(meta["lengths"])]:
            raise ValueError(f"invalid application fields for frame {index}")
        result.append((index, seed, size, int(row["timestamp"])))
    return result  # ordinary values, so retaining them holds no receive memory


class Results:
    def __init__(self, args):
        self.args, self.count, self.bytes, self.last = args, 0, 0, -1
        self.begin = time.monotonic()

    def accept(self, record, bad, total):
        index, seed, size, timestamp = record
        if bad or total != seed * size or timestamp <= 0:
            raise ValueError(f"payload mismatch for frame {index}: bad={bad}, sum={total}")
        if index <= self.last or (self.args.delivery == "every" and index != self.count):
            raise ValueError(f"unexpected frame index {index} after {self.last}")
        self.last, self.count, self.bytes = index, self.count + 1, self.bytes + size

    def finish(self, sub):
        expected = self.args.expected
        if expected is None and self.args.delivery == "every":
            expected = self.args.start
        if expected is not None and self.count != expected:
            raise ValueError(f"received {self.count} frames, expected {expected}")
        if self.count == 0:
            raise ValueError("no frames received")
        seconds = time.monotonic() - self.begin
        health = sub.health()
        if health["failure"] or health["quarantined_bytes"]:
            raise RuntimeError(f"failed or quarantined memory: {health}")
        print(json.dumps({"verified_frames": self.count, "payload_bytes": self.bytes,
                          "elapsed_seconds": seconds, "outcome": sub.outcome,
                          "transport": health["transport"], "skipped": health["skipped"]}), flush=True)


class PendingGpu:
    """Own outputs until CUDA completion; the transport owns its separate receive completion."""
    def __init__(self, capacity, stream, results):
        if capacity < 1:
            raise ValueError("--inflight must be positive")
        self.capacity, self.stream, self.results = capacity, stream, results
        self.pending = deque()

    def append(self, event, entries, retained=()):
        self.pending.append((event, entries, retained))
        if len(self.pending) >= self.capacity:
            self.drain_one()

    def drain_one(self):
        event, entries, retained = self.pending.popleft()
        event.synchronize()
        for record, bad, total in entries:
            self.results.accept(record, int(bad.get()), int(total.get()))

    def finish(self):
        while self.pending:
            self.drain_one()
