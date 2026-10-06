"""Upload pinned receive frames asynchronously, release them, then analyze on the GPU."""
import cupy as cp

from common import arguments, connect, frames, records, Results, PendingGpu


def main():
    args = arguments(__doc__, gpu=True).parse_args()
    cp.cuda.Device(args.gpu).use()
    stream = cp.cuda.Stream(non_blocking=True)
    results = Results(args)
    pending = PendingGpu(args.inflight, stream, results)
    sub, meta = connect(args, f"pinned:{args.gpu}")
    try:
        for batch in frames(sub, args):
            rows = records(batch, meta)
            outputs, entries = [], []
            with stream:
                for i, row in enumerate(rows):
                    host = batch.payload(i)
                    if host.nbytes != row[2]:
                        raise ValueError("transport length and application field disagree")
                    if cp.cuda.runtime.pointerGetAttributes(host.ctypes.data).type != cp.cuda.runtime.memoryTypeHost:
                        raise ValueError("receive allocation is not CUDA page-locked host memory")
                    output = cp.empty(host.size, dtype=cp.uint8)
                    cp.cuda.runtime.memcpyAsync(output.data.ptr, host.ctypes.data, host.nbytes,
                                                cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
                    outputs.append(output)
                # Record immediately after the last upload read. Credit need not wait for the
                # later analysis kernels, which only use independent GPU output allocations.
                batch.release_after(stream=stream.ptr)
                del host, batch
                for row, output in zip(rows, outputs):
                    entries.append((row, cp.count_nonzero(output != row[1]),
                                    output.sum(dtype=cp.uint64)))
                done = cp.cuda.Event()
                done.record(stream)
            pending.append(done, entries, outputs)
        pending.finish()
        results.finish(sub)
    finally:
        stream.synchronize()
        pending.pending.clear()
        sub.close()


if __name__ == "__main__":
    main()
