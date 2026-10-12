"""Isolate projection-to-sinogram layout copies from production per-frame ingestion."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tomography"))


@contextmanager
def region(name):
    from cupy.cuda import nvtx

    nvtx.RangePush(name)
    try:
        yield
    finally:
        nvtx.RangePop()


def measure(name, operation, stream, repeats):
    import cupy as cp

    samples = []
    with stream:
        operation()
        stream.synchronize()
        for repeat in range(repeats):
            start_event, stop_event = cp.cuda.Event(), cp.cuda.Event()
            with region(name):
                stream.synchronize()
                begin = time.perf_counter()
                start_event.record(stream)
                operation()
                stop_event.record(stream)
                stop_event.synchronize()
                samples.append(dict(wall_ms=(time.perf_counter() - begin) * 1000,
                                    stream_ms=cp.cuda.get_elapsed_time(start_event, stop_event)))
    return dict(name=name, samples=samples,
                median_wall_ms=float(np.median([sample["wall_ms"] for sample in samples])),
                median_stream_ms=float(np.median([sample["stream_ms"] for sample in samples])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--columns", type=int, default=512)
    parser.add_argument("--angles", type=int, default=360)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16, 128, 360])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.columns, args.angles, args.repeats, *args.batch) < 1:
        parser.error("dimensions, batch sizes and repeats must be positive")
    import cupy as cp
    from processors import Processor, array_at
    from reconstruction import configuration

    cp.cuda.Device(0).use()
    stream = cp.cuda.Stream(non_blocking=True)
    shape = (args.angles, args.rows, args.columns)
    host = (np.arange(np.prod(shape), dtype=np.uint32) % 10007).astype(np.float32).reshape(shape)
    expected = host.transpose(1, 0, 2)
    theta = np.linspace(0, np.pi, args.angles + 1, endpoint=False, dtype=np.float32)
    scan = dict(rows=args.rows, columns=args.columns, angles=args.angles + 1,
                theta=theta.tolist(), reconstruction=configuration("fbp", "ram-lak"))
    processor = Processor("reconstruct", scan, 0, stream.ptr, 1)
    with stream:
        frames = cp.asarray(host)
        sinogram = cp.empty((args.rows, args.angles, args.columns), dtype=cp.float32)
        projection_major = cp.empty_like(frames)
        borrowed = array_at(frames.data.ptr, frames.shape, frames.dtype, 0)
    stream.synchronize()
    assert borrowed.data.ptr == frames.data.ptr
    assert borrowed.flags.c_contiguous and not sinogram[:, 0, :].flags.c_contiguous
    frame_bytes = args.rows * args.columns * 4
    metadata = [(frames.data.ptr + projection * frame_bytes, frame_bytes, 2,
                 projection, float(theta[projection])) for projection in range(args.angles)]
    results = []
    for batch_size in args.batch:
        def copy_batch():
            for begin in range(0, args.angles, batch_size):
                stop = min(args.angles, begin + batch_size)
                cp.copyto(sinogram[:, begin:stop, :], frames[begin:stop].transpose(1, 0, 2))

        result = measure(f"layout.batch{batch_size}", copy_batch, stream, args.repeats)
        np.testing.assert_array_equal(sinogram.get(), expected)
        result.update(copy_calls=(args.angles + batch_size - 1) // batch_size,
                      extra_retained_bytes=0, validated=True)
        results.append(result)

    def contiguous_then_transpose():
        for projection in range(args.angles):
            cp.copyto(projection_major[projection], frames[projection])
        cp.copyto(sinogram, projection_major.transpose(1, 0, 2))

    result = measure("layout.contiguous_then_transpose", contiguous_then_transpose, stream, args.repeats)
    np.testing.assert_array_equal(sinogram.get(), expected)
    result.update(copy_calls=args.angles + 1, extra_retained_bytes=projection_major.nbytes, validated=True)
    results.append(result)

    def production_ingest():
        processor.begin_scan()
        for pointer, nbytes, kind, projection, angle in metadata:
            processor.consume(pointer, nbytes, 0, kind, projection, angle)

    with stream:
        processor.sinogram.fill(cp.nan)
    result = measure("production.consume_without_final_projection", production_ingest, stream, args.repeats)
    assert processor.projections == args.angles and processor.reconstruct_ns == 0
    actual = processor.sinogram.get()
    np.testing.assert_array_equal(actual[:, :args.angles, :], expected)
    assert np.isnan(actual[:, args.angles, :]).all()
    np.testing.assert_array_equal(frames.get(), host)
    result.update(copy_calls=args.angles, extra_retained_bytes=0, validated=True,
                  declared_angles=args.angles + 1, ingested_angles=args.angles)
    results.append(result)

    def ingest_group(entries):
        first_projection = entries[0][3]
        projections = np.asarray([entry[3] for entry in entries], dtype=np.int64)
        angles = np.asarray([entry[4] for entry in entries], dtype=np.float64)
        if not np.array_equal(projections, np.arange(first_projection, first_projection + len(entries))):
            raise ValueError("group must have consecutive projections")
        if first_projection < 0 or projections[-1] >= len(theta):
            raise ValueError("projection outside declared scan")
        if not np.all(np.isclose(angles, theta[projections])):
            raise ValueError("projection angle changed")
        for offset, (pointer, nbytes, kind, _, _) in enumerate(entries):
            if kind != 2 or nbytes != frame_bytes:
                raise ValueError("group must contain complete corrected projections")
            if pointer != entries[0][0] + offset * frame_bytes:
                raise ValueError("group crosses a receive allocation boundary")
        source = array_at(entries[0][0], (len(entries), args.rows, args.columns), cp.float32, 0)
        cp.copyto(processor.sinogram[:, first_projection:first_projection + len(entries), :],
                  source.transpose(1, 0, 2))
        processor.projections += len(entries)

    for batch_size in args.batch:
        def grouped_ingest():
            processor.begin_scan()
            for begin in range(0, args.angles, batch_size):
                ingest_group(metadata[begin:begin + batch_size])

        result = measure(f"experimental.grouped_ingest{batch_size}", grouped_ingest, stream, args.repeats)
        assert processor.projections == args.angles and processor.reconstruct_ns == 0
        np.testing.assert_array_equal(processor.sinogram.get()[:, :args.angles, :], expected)
        result.update(copy_calls=(args.angles + batch_size - 1) // batch_size,
                      extra_retained_bytes=0, validated=True)
        results.append(result)

    invalid = list(metadata[:min(2, args.angles)])
    pointer, nbytes, kind, projection, angle = invalid[0]
    invalid[0] = (pointer, nbytes, kind, projection, angle + 1)
    count_before = processor.projections
    before = processor.sinogram.get()
    try:
        with stream:
            ingest_group(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("changed angle was accepted")
    assert processor.projections == count_before
    np.testing.assert_array_equal(processor.sinogram.get(), before)
    processor.drain()
    rendered = dict(shape=shape, dtype="float32", cupy=cp.__version__, results=results,
                    zero_copy_borrowed_pointer=True,
                    scope="All input/output allocations preallocated on one GPU. "
                          "Layout-only variants omit production validation and pointer-view construction. "
                          "Production receives one fewer projection than its declared scan, avoiding FBP. "
                          "Pointer and metadata tuples are prepared outside timing for both ingestion paths. "
                          "Experimental grouped ingestion validates kinds, lengths, projection continuity, "
                          "angles and contiguous pointers before a single borrowed-view layout copy. "
                          "CUDA event intervals include host submission gaps, not just kernel execution. "
                          "Batch variants assume contiguous retained receive frames and require caller integration.")
    args.output.write_text(json.dumps(rendered, indent=2) + "\n")
    print(json.dumps(rendered), flush=True)


if __name__ == "__main__":
    main()
