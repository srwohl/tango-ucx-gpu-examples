"""Average streamed dark/flat frames into owned GPU arrays, then correct data on that GPU."""
from collections import deque
import json
from pathlib import Path

import cupy as cp
import numpy as np

from common import arguments, connect, frames
from dark_flat_reference import Reference


class Calibration:
    """One allocation per map; all updates and reads use the caller's CUDA stream."""
    def __init__(self, shape, meta, minimum):
        self.meta, self.minimum = meta, minimum
        self.dark_sum = cp.empty(shape, dtype=cp.float32)
        self.flat_sum = cp.empty_like(self.dark_sum)
        self.dark = cp.empty_like(self.dark_sum)
        self.flat = cp.empty_like(self.dark_sum)
        self.inverse = cp.empty_like(self.dark_sum)
        self.id, self.darks, self.flats, self.data = 0, 0, 0, 0
        self.correct_kernel = cp.ElementwiseKernel(
            "uint16 raw, float32 dark, float32 inverse", "float32 out",
            'out = raw == 65535 ? nanf("") : (float(raw) - dark) * inverse;',
            "streamed_dark_flat_correct")
        self.prepare_kernel = cp.ElementwiseKernel(
            "float32 flat, float32 dark, float32 minimum", "float32 inverse",
            'float response = flat - dark; inverse = response > minimum ? 1.0f / response : nanf("");',
            "streamed_dark_flat_prepare")

    def consume(self, raw, kind, calibration_id, sample, output):
        if calibration_id != self.id:
            if calibration_id != self.id + 1 or kind != 0 or sample != 0 or (self.id and not self.data):
                raise ValueError("calibration change must begin with dark frames after data")
            self.id, self.darks, self.flats, self.data = calibration_id, 0, 0, 0
            # The previous calibration's kernels precede these fills on the same GPU stream.
            self.dark_sum.fill(0)
            self.flat_sum.fill(0)
        if kind == 0:
            if self.flats or self.data or sample != self.darks or self.darks >= self.meta["dark_frames"]:
                raise ValueError("unexpected dark frame")
            cp.add(self.dark_sum, raw, out=self.dark_sum)
            self.darks += 1
            if self.darks == self.meta["dark_frames"]:
                cp.divide(self.dark_sum, self.darks, out=self.dark)
        elif kind == 1:
            if self.darks != self.meta["dark_frames"] or self.data or sample != self.flats or self.flats >= self.meta["flat_frames"]:
                raise ValueError("unexpected flat frame")
            cp.add(self.flat_sum, raw, out=self.flat_sum)
            self.flats += 1
            if self.flats == self.meta["flat_frames"]:
                cp.divide(self.flat_sum, self.flats, out=self.flat)
                self.prepare_kernel(self.flat, self.dark, cp.float32(self.minimum), self.inverse)
        elif kind == 2:
            if self.darks != self.meta["dark_frames"] or self.flats != self.meta["flat_frames"] or sample != self.data:
                raise ValueError("data arrived before complete matching calibration")
            self.correct_kernel(raw, self.dark, self.inverse, output)
            self.data += 1
        else:
            raise ValueError(f"unknown frame kind {kind}")


def main():
    parser = arguments(__doc__, gpu=True)
    parser.add_argument("--receive", choices=("gpu", "pinned"), default="gpu")
    parser.add_argument("--minimum-response", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="new directory for calibration maps and last corrected frame")
    args = parser.parse_args()
    if args.delivery != "every":
        parser.error("streamed calibration uses every delivery so no calibration frame is skipped")
    if args.inflight < 1 or not np.isfinite(args.minimum_response) or args.minimum_response < 0:
        parser.error("--inflight must be positive and --minimum-response finite and nonnegative")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    cp.cuda.Device(args.gpu).use()
    stream = cp.cuda.Stream(non_blocking=True)
    memory = f"cuda:{args.gpu}" if args.receive == "gpu" else f"pinned:{args.gpu}"
    sub, meta = connect(args, memory, example="dark-flat-v1")
    pending = deque()
    count, data_count, pixel_count, last_error = 0, 0, 0, 0.0
    try:
        shape = tuple(sub.description["shape"])
        if len(shape) != 2 or sub.description["element"] != "u16" or meta["pattern"] != "spatial-u16-v1":
            raise ValueError("expected the simulated uint16 calibration frame pattern")
        reference = Reference(shape, meta, args.minimum_response)
        with stream:
            calibration = Calibration(shape, meta, args.minimum_response)
            # Application-owned arrays, distinct from the UCX receive ring. Reuse each slot
            # only after its completion event and verification, keeping allocation bounded.
            outputs = cp.empty((args.inflight, args.batch, *shape), dtype=cp.float32)
            uploads = cp.empty((args.inflight, args.batch, *shape), dtype=cp.uint16) if args.receive == "pinned" else None
        pointers = [calibration.dark.data.ptr, calibration.flat.data.ptr, calibration.inverse.data.ptr]
        print(json.dumps({"owned_calibration_gpu_bytes": 5 * int(np.prod(shape)) * 4,
                          "shape": shape, "receive": args.receive, "inflight": args.inflight}), flush=True)
        submitted = 0

        def drain():
            nonlocal data_count, pixel_count, last_error
            event, slot, rows, snapshots = pending.popleft()
            event.synchronize()
            for i, row in enumerate(rows):
                index, kind, calibration_id, sample = row
                if kind != 2:
                    continue
                actual = outputs[slot, i].get()
                expected = reference.corrected(calibration_id, sample)
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7, equal_nan=True,
                                           err_msg=f"frame {index}, calibration {calibration_id}")
                valid = np.isfinite(expected)
                if np.any(valid):
                    last_error = max(last_error, float(np.max(np.abs(actual[valid] - expected[valid]))))
                data_count += 1
                pixel_count += actual.size
                if args.output:
                    np.save(args.output / "last_corrected.npy", actual)
            for calibration_id, dark, flat, inverse in snapshots:
                expected_dark, expected_flat = reference.calibration(calibration_id)
                np.testing.assert_allclose(dark.get(), expected_dark, rtol=0, atol=0)
                np.testing.assert_allclose(flat.get(), expected_flat, rtol=0, atol=0)
                if args.output:
                    np.savez(args.output / f"calibration-{calibration_id}.npz",
                             dark=dark.get(), flat=flat.get(), inverse=inverse.get())

        for batch in frames(sub, args):
            if len(pending) == args.inflight:
                drain()
            slot = submitted % args.inflight
            submitted += 1
            rows = [(int(r["index"]), int(r["kind"]), int(r["calibration_id"]), int(r["sample"]))
                    for r in batch.records]
            for row in rows:
                if row[0] != count:
                    raise ValueError("missing or out-of-order calibration frame")
                count += 1
            snapshots = []

            def process(images):
                for i, row in enumerate(rows):
                    _, kind, calibration_id, sample = row
                    calibration.consume(images[i], kind, calibration_id, sample, outputs[slot, i])
                    if kind == 1 and sample + 1 == meta["flat_frames"]:
                        # Capture a GPU-owned snapshot before the next calibration overwrites
                        # these maps. These copies do not retain any receive batch.
                        snapshots.append((calibration_id, calibration.dark.copy(),
                                          calibration.flat.copy(), calibration.inverse.copy()))

            with stream:
                if args.receive == "gpu":
                    with batch.gpu_view(stream=stream.ptr) as view:
                        images = cp.from_dlpack(view)
                        process(images)
                        del images
                    del view
                else:
                    host = batch.array
                    if cp.cuda.runtime.pointerGetAttributes(host.ctypes.data).type != cp.cuda.runtime.memoryTypeHost:
                        raise ValueError("receive buffer is not CUDA page-locked")
                    cp.cuda.runtime.memcpyAsync(uploads[slot].data.ptr, host.ctypes.data, host.nbytes,
                                                cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
                    batch.release_after(stream=stream.ptr)  # the final host upload read is queued
                    del host, batch
                    process(uploads[slot, :len(rows)])
                done = cp.cuda.Event()
                done.record(stream)
            if args.receive == "gpu":
                del batch
            pending.append((done, slot, rows, snapshots))
        while pending:
            drain()
        expected_data = args.expected
        if expected_data is None and args.start is not None:
            expected_data = args.start * meta["calibrations"]
        if expected_data is not None and data_count != expected_data:
            raise ValueError(f"corrected {data_count} data frames, expected {expected_data}")
        if not data_count or calibration.id != meta["calibrations"] or not calibration.data:
            raise ValueError("stream ended with incomplete calibration or no data")
        if pointers != [calibration.dark.data.ptr, calibration.flat.data.ptr, calibration.inverse.data.ptr]:
            raise ValueError("calibration allocations unexpectedly changed")
        if sub.health()["failure"] or sub.health()["quarantined_bytes"]:
            raise RuntimeError(f"receive failure: {sub.health()}")
        print(json.dumps({"verified_frames": count, "corrected_data_frames": data_count,
                          "calibrations": calibration.id, "verified_pixels": pixel_count,
                          "max_absolute_error": last_error, "outcome": sub.outcome,
                          "transport": sub.health()["transport"]}), flush=True)
    finally:
        stream.synchronize()
        pending.clear()
        sub.close()


if __name__ == "__main__":
    main()
