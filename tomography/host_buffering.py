"""Bounded scan retention and CUDA-pinned staging, independent of Tango services.

Only staging is pinned. Retained scans use ordinary host RAM; CUDA completion
events protect every staging slot before the CPU or GPU reuses it.
"""
import numpy as np


TRANSFER_SLOTS = 2


def memory_plan(scan, reconstruction):
    """Validate launch budgets and a scan's block layout before allocating anything."""
    config = scan.get("buffering", {})
    mode = config.get("sinogram_memory", "gpu")
    if mode not in ("gpu", "host"):
        raise ValueError("sinogram_memory must be gpu or host")
    rows, angles, columns = (int(scan[key]) for key in ("rows", "angles", "columns"))
    if min(rows, angles, columns) < 1:
        raise ValueError("scan dimensions must be positive")
    size = reconstruction.get("slices_per_block", 0)
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 0:
        raise ValueError("slices_per_block must be a nonnegative integer")
    block_rows = min(size or rows, rows)
    output_mode = config.get("output_mode", "volume")
    if output_mode not in ("volume", "blocks"):
        raise ValueError("output_mode must be volume or blocks")
    output_rows = rows
    if output_mode == "blocks":
        output_rows = config.get("output_block_rows", block_rows)
        if isinstance(output_rows, bool) or not isinstance(output_rows, int) or not 1 <= output_rows <= rows:
            raise ValueError("output_block_rows must lie within the scan")
        if not size or block_rows > output_rows:
            raise ValueError("block output requires positive slices_per_block within launch output capacity")
        if output_rows * columns * columns * 4 > 2 * 1024**3:
            raise ValueError("output block exceeds the transport's 2 GiB frame limit")
        host_output = rows * columns * columns * 4
        host_budget = config.get("output_host_budget_bytes", 1024 * 1024**2)
        if isinstance(host_budget, bool) or not isinstance(host_budget, int) or host_budget < host_output:
            raise ValueError(f"output_host_budget_bytes requires {host_output} bytes")
    sinogram_bytes = rows * angles * columns * 4
    frame_bytes = rows * columns * 4
    block_input_bytes = block_rows * angles * columns * 4
    block_output_bytes = block_rows * columns * columns * 4
    cpu = reconstruction["algorithm"] == "gridrec"
    pinned_bytes = TRANSFER_SLOTS * (frame_bytes + (block_output_bytes if cpu else block_input_bytes))
    if mode == "host":
        for key, required in (("host_budget_bytes", sinogram_bytes),
                              ("pinned_budget_bytes", pinned_bytes)):
            budget = config.get(key, (1024 if key == "host_budget_bytes" else 128) * 1024**2)
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
                raise ValueError(f"{key} must be a positive integer")
            if required > budget:
                raise ValueError(f"{key} requires {required} bytes, exceeds {budget}; "
                                 "increase the budget or reduce slices_per_block")
    return dict(sinogram_memory=mode, block_rows=block_rows, transfer_slots=TRANSFER_SLOTS,
                retained_host_bytes=sinogram_bytes if mode == "host" else 0,
                pinned_staging_bytes=pinned_bytes if mode == "host" else 0,
                gpu_sinogram_bytes=0 if mode == "host" else sinogram_bytes,
                gpu_block_input_bytes=TRANSFER_SLOTS * block_input_bytes if mode == "host" and not cpu else 0,
                output_mode=output_mode, output_block_rows=output_rows,
                host_output_bytes=rows * columns * columns * 4 if output_mode == "blocks" else 0,
                gpu_output_bytes=output_rows * columns * columns * 4,
                gpu_output_ring_budget_bytes=max(config.get("transport_budget_bytes", 256 << 10),
                    output_rows * columns * columns * 4 * 8 + (192 << 10)))


def pinned_empty(cp, shape):
    """NumPy retains the allocation owner through its buffer/base chain."""
    allocation = cp.cuda.alloc_pinned_memory(int(np.prod(shape)) * 4)
    return np.ndarray(shape, np.float32, buffer=allocation)


class HostScanBuffer:
    """Retain exactly one scan; two pinned frames overlap arrivals and host packing.

    The caller must queue downloads on the receive view's owning stream so its
    transport completion event follows the last read of the borrowed GPU frame.
    """

    def __init__(self, scan, cp, stream):
        self.data = np.empty((scan["rows"], scan["angles"], scan["columns"]), np.float32)
        shape = (scan["rows"], scan["columns"])
        self.frames = [pinned_empty(cp, shape) for _ in range(TRANSFER_SLOTS)]
        self.events = [cp.cuda.Event(disable_timing=True) for _ in self.frames]
        self.pending = [None] * TRANSFER_SLOTS
        self.stream = stream
        self.received = 0

    def _drain(self, slot):
        projection = self.pending[slot]
        if projection is not None:
            self.events[slot].synchronize()
            # Projection -> sinogram layout conversion: one explicit CPU copy.
            np.copyto(self.data[:, projection, :], self.frames[slot])
            self.pending[slot] = None

    def begin_scan(self):
        self.finish_downloads()
        self.received = 0

    def append(self, source, projection):
        if projection != self.received or projection >= self.data.shape[1]:
            raise ValueError("host buffer expects projections in order")
        slot = projection % TRANSFER_SLOTS
        self._drain(slot)
        source.get(out=self.frames[slot], stream=self.stream, blocking=False)
        self.events[slot].record(self.stream)
        self.pending[slot] = projection
        self.received += 1

    def finish_downloads(self):
        for slot in range(TRANSFER_SLOTS):
            self._drain(slot)


class BlockTransfers:
    """Two reusable pinned slots for GPU input or CPU reconstruction output.

    GPU input prefetch uses a separate stream. The consumer must call consumed()
    after its last read, and finish() before releasing scratch or on any error.
    """

    def __init__(self, scan, plan, cp, compute_stream, *, cpu):
        self.compute_stream = compute_stream
        self.cpu = cpu
        rows = plan["block_rows"]
        shape = (rows, scan["columns"], scan["columns"]) if cpu else (
            rows, scan["angles"], scan["columns"])
        self.host = [pinned_empty(cp, shape) for _ in range(TRANSFER_SLOTS)]
        self.ready = [cp.cuda.Event(disable_timing=True) for _ in self.host]
        self.done = [cp.cuda.Event(disable_timing=True) for _ in self.host]
        self.used = [False] * TRANSFER_SLOTS
        self.transfer_stream = None if cpu else cp.cuda.Stream(non_blocking=True)
        if not cpu:
            with self.transfer_stream:
                self.device = [cp.empty(shape, dtype=cp.float32) for _ in self.host]

    def _reuse(self, slot):
        if self.used[slot]:
            self.done[slot].synchronize()

    def prefetch(self, source, slot):
        self._reuse(slot)
        rows = source.shape[0]
        np.copyto(self.host[slot][:rows], source)
        self.device[slot][:rows].set(self.host[slot][:rows], stream=self.transfer_stream)
        self.ready[slot].record(self.transfer_stream)

    def input(self, slot, rows):
        self.compute_stream.wait_event(self.ready[slot])
        return self.device[slot][:rows]

    def consumed(self, slot):
        self.done[slot].record(self.compute_stream)
        self.used[slot] = True

    def upload_result(self, result, output, slot):
        self._reuse(slot)
        rows = result.shape[0]
        # TomoPy owns its returned array; one explicit copy into reusable pinned staging.
        np.copyto(self.host[slot][:rows], result)
        output.set(self.host[slot][:rows], stream=self.compute_stream)
        self.consumed(slot)

    def finish(self):
        # Also protects buffers when a kernel/API raises before consumed() is recorded.
        if self.transfer_stream is not None:
            self.transfer_stream.synchronize()
        self.compute_stream.synchronize()
