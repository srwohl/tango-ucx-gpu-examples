"""Experiment-only grouped assembly into bounded, independently owned scan buffers."""
import time

import cupy as cp
import numpy as np

from processors import Processor, array_at


class ScanAssembly:
    def __init__(self, scan, gpu, assembly_pointer, reconstruction_pointer, iterations, buffers):
        if buffers not in (1, 2):
            raise ValueError("the experiment supports one or two scan buffers")
        buffering = scan.get("buffering", {})
        if buffering.get("sinogram_memory", "gpu") != "gpu" or buffering.get("output_mode", "volume") != "volume":
            raise ValueError("the experiment requires GPU sinograms and whole-volume output")
        if scan["reconstruction"]["algorithm"] != "fbp":
            raise ValueError("the experiment supports FBP only")
        self.scan, self.gpu = scan, gpu
        self.assembly = cp.cuda.ExternalStream(assembly_pointer)
        self.reconstruction = cp.cuda.ExternalStream(reconstruction_pointer)
        self.processors = [Processor("reconstruct", scan, gpu, assembly_pointer, iterations)
                           for _ in range(buffers)]
        self.identities = [None] * buffers
        self.frame_bytes = scan["rows"] * scan["columns"] * 4
        self.theta = np.asarray(scan["theta"], dtype=np.float32)

    def begin_scan(self, buffer, scan_id, options):
        if options["algorithm"] != "fbp":
            raise ValueError("the experiment supports FBP only")
        processor = self.processors[buffer]
        processor.begin_scan()
        processor.configure_reconstruction(options)
        self.identities[buffer] = scan_id

    def consume_group(self, buffer, frames):
        processor = self.processors[buffer]
        if not frames or self.identities[buffer] is None:
            raise ValueError("group requires an active scan and at least one projection")
        first = processor.projections
        stop = first + len(frames)
        if stop > self.scan["angles"]:
            raise ValueError("group extends beyond the scan")
        projections = np.asarray([frame[3] for frame in frames], dtype=np.int64)
        angles = np.asarray([frame[4] for frame in frames], dtype=np.float64)
        if not np.array_equal(projections, np.arange(first, stop)):
            raise ValueError("group must have consecutive projections")
        if not np.all(np.isclose(angles, self.theta[first:stop])):
            raise ValueError("projection angle changed")
        for offset, (pointer, nbytes, kind, _, _) in enumerate(frames):
            if kind != 2 or nbytes != self.frame_bytes:
                raise ValueError("group must contain complete corrected projections")
            if pointer != frames[0][0] + offset * self.frame_bytes:
                raise ValueError("group crosses a receive allocation boundary")
        with cp.cuda.Device(self.gpu), self.assembly:
            cp.cuda.nvtx.RangePush(f"overlap.assemble.scan{self.identities[buffer]}")
            try:
                source = array_at(frames[0][0], (len(frames), self.scan["rows"], self.scan["columns"]),
                                  cp.float32, self.gpu)
                cp.copyto(processor.sinogram[:, first:stop, :], source.transpose(1, 0, 2))
                processor.projections = stop
            finally:
                cp.cuda.nvtx.RangePop()

    def reconstruct(self, buffer, output_pointer):
        processor = self.processors[buffer]
        processor.finish()
        output = array_at(output_pointer, (self.scan["rows"], self.scan["columns"], self.scan["columns"]),
                          cp.float32, self.gpu)
        with cp.cuda.Device(self.gpu), self.reconstruction:
            cp.cuda.nvtx.RangePush(f"overlap.reconstruct.scan{self.identities[buffer]}")
            begin = time.perf_counter_ns()
            processor.stream = self.reconstruction
            try:
                processor.reconstruct(output)
                self.reconstruction.synchronize()
                return time.perf_counter_ns() - begin
            finally:
                processor.stream = self.assembly
                cp.cuda.nvtx.RangePop()

    def drain(self):
        with cp.cuda.Device(self.gpu):
            self.assembly.synchronize()
            self.reconstruction.synchronize()

