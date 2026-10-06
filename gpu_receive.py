"""Analyze opaque GPU frames through DLPack or the external-adapter pointer interface."""
import cupy as cp

from common import arguments, connect, frames, records, Results, PendingGpu


def main():
    parser = arguments(__doc__, gpu=True)
    parser.add_argument("--access", choices=("dlpack", "pointer"), default="dlpack")
    args = parser.parse_args()
    cp.cuda.Device(args.gpu).use()
    stream = cp.cuda.Stream(non_blocking=True)
    results = Results(args)
    pending = PendingGpu(args.inflight, stream, results)
    sub, meta = connect(args, f"cuda:{args.gpu}")
    try:
        for batch in frames(sub, args):
            rows = records(batch, meta)
            entries, tensors = [], []
            with stream, batch.gpu_view(stream=stream.ptr) as view:
                for i, row in enumerate(rows):
                    payload = view.payload(i)
                    if payload.nbytes != row[2]:
                        raise ValueError("transport length and application field disagree")
                    if args.access == "dlpack":
                        tensor = cp.from_dlpack(payload)
                        tensors.append(tensor)  # each imported tensor holds the receive batch
                    else:
                        memory = cp.cuda.UnownedMemory(payload.ptr, payload.nbytes, view,
                                                       device_id=args.gpu)
                        tensor = cp.ndarray((payload.nbytes,), dtype=cp.uint8,
                                            memptr=cp.cuda.MemoryPointer(memory, 0))
                    entries.append((row, cp.count_nonzero(tensor != row[1]),
                                    tensor.sum(dtype=cp.uint64)))
                # All last reads are now queued on the tracked GPU stream.
                done = cp.cuda.Event()
                done.record(stream)
            # Pointer mode drops every receive reference before the GPU finishes. The event
            # recorded by closing the GPU view prevents Credit and allocation reuse.
            del tensor, payload, view, batch
            pending.append(done, entries, tensors)
            del tensors
        pending.finish()
        results.finish(sub)
    finally:
        stream.synchronize()
        pending.pending.clear()  # release imported tensors while their tracked stream still exists
        sub.close()


if __name__ == "__main__":
    main()
