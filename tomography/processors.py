"""GPU processing and selectable reconstruction; C++ owns transport allocations."""
import cupy as cp
import numpy as np
from nvidia import nvcomp

from host_buffering import BlockTransfers, HostScanBuffer, memory_plan
from reconstruction import SirtGPU, configuration, fbp_gpu, gridrec, postprocess, slice_blocks


def array_at(pointer, shape, dtype, gpu):
    """View memory whose completion and ownership C++ controls."""
    dtype = cp.dtype(dtype)
    memory = cp.cuda.UnownedMemory(pointer, int(np.prod(shape)) * dtype.itemsize, None,
                                  device_id=gpu)
    return cp.ndarray(shape, dtype=dtype, memptr=cp.cuda.MemoryPointer(memory, 0))


class Processor:
    def __init__(self, role, scan, gpu, stream_pointer, iterations):
        self.role, self.scan, self.gpu = role, scan, gpu
        self.reconstruction = scan.get("reconstruction", configuration("sirt", iterations=iterations))
        self.shape = (scan["rows"], scan["columns"])
        element = scan.get("element", "u16")
        if element not in ("u16", "f32"):
            raise ValueError("detector element must be u16 or f32")
        self.detector_dtype = cp.dtype("uint16" if element == "u16" else "float32")
        self.detector_frame_bytes = int(np.prod(self.shape)) * self.detector_dtype.itemsize
        cp.cuda.Device(gpu).use()
        self.stream = cp.cuda.ExternalStream(stream_pointer)
        self.projections = 0
        self.dark_seen = self.flat_seen = False
        self.theta = np.asarray(scan["theta"], dtype=np.float32)
        self.begin_scan()
        with self.stream:
            if role == "decompress":
                self.codec = nvcomp.Codec(algorithm="LZ4", cuda_stream=stream_pointer,
                                         bitstream_kind=nvcomp.BitstreamKind.RAW, device_id=gpu)
                self.decoding = self.codec.decompression_config(
                    self.codec.compression_config(self.detector_frame_bytes))
                # Packed frame pointers may be unaligned: use aligned, reusable GPU scratch.
                self.compressed = cp.empty(scan["max_compressed_bytes"], dtype=cp.uint8)
            elif role == "correct":
                self.dark = cp.empty(self.shape, dtype=cp.float32)
                self.flat = cp.empty_like(self.dark)
                self.correct = cp.ElementwiseKernel(
                    f"{self.detector_dtype.name} raw, float32 dark, float32 flat", "float32 attenuation",
                    "float t = (float(raw) - dark) / (flat - dark); "
                    "attenuation = -logf(fminf(1.0f, fmaxf(1e-6f, t)));",
                    f"tomography_dark_flat_log_{element}")
            else:
                self.buffer_plan = memory_plan(scan, self.reconstruction)
                if self.buffer_plan["sinogram_memory"] == "host":
                    self.host_buffer = HostScanBuffer(scan, cp, self.stream)
                    self.sinogram = self.host_buffer.data
                    self.configure_reconstruction(self.reconstruction)
                else:
                    self.sinogram = cp.empty((scan["rows"], scan["angles"], scan["columns"]),
                                             dtype=cp.float32)

    def begin_scan(self):
        if hasattr(self, "host_buffer"):
            self.host_buffer.begin_scan()
        self.projections = self.received = 0
        self.output_next_start = 0
        self.dark_seen = self.flat_seen = False

    def configure_reconstruction(self, options):
        """Called by the worker only at a scan boundary with validated settings."""
        if self.role != "reconstruct":
            raise ValueError("only the reconstruction stage accepts settings")
        plan = memory_plan(self.scan, options)
        rows = self.scan["rows"]
        old_size = min(self.reconstruction.get("slices_per_block", 0) or rows, rows)
        new_size = min(options.get("slices_per_block", 0) or rows, rows)
        if hasattr(self, "host_buffer"):
            layout = (plan["block_rows"], options["algorithm"] == "gridrec")
            if getattr(self, "transfer_layout", None) != layout or not hasattr(self, "transfers"):
                if hasattr(self, "transfers"):
                    self.transfers.finish()
                    del self.transfers
                # Release the old pool before allocating its replacement so owned
                # pinned staging stays within the new plan. If allocation fails,
                # invalidate the layout so a later configuration can retry safely.
                self.transfer_layout = None
                self.transfers = BlockTransfers(self.scan, plan, cp, self.stream, cpu=layout[1])
                self.transfer_layout = layout
        if old_size != new_size and hasattr(self, "sirt"):
            # Only keep scratch for the current block layout, not every live setting.
            del self.sirt
        self.buffer_plan = plan
        self.reconstruction = dict(options)

    def _gpu_reconstruct_block(self, sinogram, volume):
        options = self.reconstruction
        if options["algorithm"] == "sirt":
            if not hasattr(self, "sirt"):
                self.sirt = {}
            rows = sinogram.shape[0]
            if rows not in self.sirt:
                self.sirt[rows] = SirtGPU(sinogram.shape, self.theta, self.gpu)
            self.sirt[rows].run(sinogram, volume, options["iterations"],
                               relaxation=options.get("relaxation", 1.0),
                               min_constraint=options.get("min_constraint", 0.0),
                               max_constraint=options.get("max_constraint"))
        elif options["algorithm"] == "fbp":
            fbp_gpu(sinogram, volume, self.theta, self.gpu, options["filter"],
                    filter_cutoff=options.get("filter_cutoff"))
        else:
            raise ValueError(f"unknown GPU reconstruction algorithm: {options['algorithm']}")

    def _reconstruct_host(self, output):
        options = self.reconstruction
        self.host_buffer.finish_downloads()
        blocks = list(slice_blocks(self.scan["rows"], options.get("slices_per_block", 0)))
        try:
            if options["algorithm"] == "gridrec":
                for index, block in enumerate(blocks):
                    result = gridrec(self.sinogram[block], self.theta, options["filter"],
                                     options["threads"], center=options.get("center"))
                    self.transfers.upload_result(result, output[block], index % 2)
            else:
                self.transfers.prefetch(self.sinogram[blocks[0]], 0)
                for index, block in enumerate(blocks):
                    slot = index % 2
                    sinogram = self.transfers.input(slot, block.stop - block.start)
                    if index + 1 < len(blocks):
                        # Queue the next H2D before computing this block. ASTRA's current
                        # device-wide handoffs can limit overlap; no speedup is assumed.
                        self.transfers.prefetch(self.sinogram[blocks[index + 1]], (index + 1) % 2)
                    self._gpu_reconstruct_block(sinogram, output[block])
                    self.transfers.consumed(slot)
        finally:
            self.transfers.finish()

    def reconstruct(self, output):
        """Fill one borrowed publisher volume using contiguous, independent z blocks.

        The full sinogram (host or GPU) and GPU output remain allocated. SIRT retains workspaces only
        for the regular block and optional shorter tail; no borrowed pointer is cached.
        """
        options = self.reconstruction
        method = options["algorithm"]
        if hasattr(self, "host_buffer"):
            self._reconstruct_host(output)
            postprocess(output, options, gpu=True)
            return
        for block in slice_blocks(self.scan["rows"], options.get("slices_per_block", 0)):
            sinogram, volume = self.sinogram[block], output[block]
            if method in ("sirt", "fbp"):
                self._gpu_reconstruct_block(sinogram, volume)
            elif method == "gridrec":
                # CPU staging is limited to one block; keep its result until upload completes.
                host = sinogram.get(stream=self.stream)
                result = gridrec(host, self.theta, options["filter"], options["threads"],
                                 center=options.get("center"))
                volume.set(result, stream=self.stream)
                self.stream.synchronize()
                del host, result
            else:
                raise ValueError(f"unknown reconstruction algorithm: {method}")
        # Gaussian smoothing mixes z neighbors: apply it once after assembling every block.
        postprocess(output, options, gpu=True)

    def reconstruct_block(self, output_pointer, start, count):
        """Write one unprocessed z block directly into a borrowed publisher slot.

        Completion/lifetime belongs to C++; whole-volume smoothing and scaling
        belongs to the host collector in bounded output mode. Zero tail padding
        because the stream description has a fixed launch-time payload capacity.
        """
        if self.projections != self.scan["angles"]:
            raise ValueError("block reconstruction requires a complete scan")
        plan = self.buffer_plan
        if plan["output_mode"] != "blocks" or not 0 < count <= plan["block_rows"]:
            raise ValueError("invalid output block")
        if not 0 <= start < start + count <= self.scan["rows"]:
            raise ValueError("output block outside scan")
        if start != self.output_next_start or count != min(plan["block_rows"], self.scan["rows"] - start):
            raise ValueError("output blocks must be produced sequentially with the configured layout")
        with self.stream:
            output = array_at(output_pointer, (plan["output_block_rows"], self.scan["columns"],
                              self.scan["columns"]), cp.float32, self.gpu)
            output[count:].fill(0)
            block = slice(start, start + count)
            try:
                if hasattr(self, "host_buffer"):
                    self.host_buffer.finish_downloads()
                    slot = (start // plan["block_rows"]) % 2
                    if self.reconstruction["algorithm"] == "gridrec":
                        result = gridrec(self.sinogram[block], self.theta, self.reconstruction["filter"],
                                         self.reconstruction["threads"], center=self.reconstruction.get("center"))
                        self.transfers.upload_result(result, output[:count], slot)
                    else:
                        if start == 0:
                            self.transfers.prefetch(self.sinogram[block], slot)
                        sinogram = self.transfers.input(slot, count)
                        if block.stop < self.scan["rows"]:
                            next_block = slice(block.stop, min(block.stop + plan["block_rows"], self.scan["rows"]))
                            self.transfers.prefetch(self.sinogram[next_block], (slot + 1) % 2)
                        self._gpu_reconstruct_block(sinogram, output[:count])
                        self.transfers.consumed(slot)
                elif self.reconstruction["algorithm"] == "gridrec":
                    result = gridrec(self.sinogram[block].get(stream=self.stream), self.theta,
                                     self.reconstruction["filter"], self.reconstruction["threads"],
                                     center=self.reconstruction.get("center"))
                    output[:count].set(result, stream=self.stream)
                    self.stream.synchronize()
                else:
                    self._gpu_reconstruct_block(self.sinogram[block], output[:count])
                self.output_next_start += count
            except Exception:
                if hasattr(self, "transfers"):
                    self.transfers.finish()
                raise

    def consume(self, pointer, nbytes, output_pointer, kind, projection, theta):
        with self.stream:
            if self.role == "decompress":
                expected_kind = self.received if self.received < 2 else 2
                expected_projection = max(0, self.received - 2)
                if kind != expected_kind or projection != expected_projection:
                    raise ValueError("missing or out-of-order compressed scan frame")
                if not 0 < nbytes <= self.compressed.nbytes:
                    raise ValueError("compressed length outside the declared capacity")
                source = array_at(pointer, (nbytes,), cp.uint8, self.gpu)
                compressed = self.compressed[:nbytes]
                cp.copyto(compressed, source)
                # Raw LZ4 is configured in bytes. Present the borrowed publisher
                # allocation as uint8 so nvCOMP sees byte capacity rather than
                # the smaller uint16/float32 element count. Correction later
                # views these same bytes using the detector dtype from the scan.
                output = array_at(output_pointer, (self.detector_frame_bytes,), cp.uint8, self.gpu)
                self.codec.decode(nvcomp.as_array(compressed, cuda_stream=self.stream.ptr),
                                  out=output, decompression_config=self.decoding)
                self.received += 1
                return
            dtype = self.detector_dtype if self.role == "correct" else cp.float32
            if nbytes != int(np.prod(self.shape)) * cp.dtype(dtype).itemsize:
                raise ValueError("upstream array length differs from the scan")
            source = array_at(pointer, self.shape, dtype, self.gpu)
            if kind == 2 and (not 0 <= projection < len(self.theta) or
                              not np.isclose(theta, self.theta[projection])):
                raise ValueError("projection index or angle changed")
            if self.role == "correct":
                if kind == 0 and not self.dark_seen and not self.flat_seen:
                    cp.copyto(self.dark, source, casting="unsafe")
                    self.dark_seen = True
                    return
                if kind == 1 and self.dark_seen and not self.flat_seen:
                    cp.copyto(self.flat, source, casting="unsafe")
                    self.flat_seen = True
                    return
                if kind != 2 or not self.flat_seen:
                    raise ValueError("projection arrived without matching dark and flat")
                if projection != self.projections or projection >= len(self.theta):
                    raise ValueError("missing or out-of-order projection")
                output = array_at(output_pointer, self.shape, cp.float32, self.gpu)
                self.correct(source, self.dark, self.flat, output)
            else:
                if kind != 2 or projection != self.projections or projection >= len(self.theta):
                    raise ValueError("reconstruction expects all corrected projections in order")
                if hasattr(self, "host_buffer"):
                    self.host_buffer.append(source, projection)
                else:
                    cp.copyto(self.sinogram[:, projection, :], source)
                if projection + 1 == self.scan["angles"] and self.buffer_plan["output_mode"] != "blocks":
                    output = array_at(output_pointer,
                                      (self.scan["rows"], self.scan["columns"], self.scan["columns"]),
                                      cp.float32, self.gpu)
                    self.reconstruct(output)
            self.projections += 1

    def consume_many(self, frames):
        """Consume one ordered, same-scan group; batch only independent LZ4 frames.

        C++ retains input/output allocations through queued stream work. No borrowed
        arrays are cached here; owned aligned scratch is reused on this same stream.
        """
        if not frames:
            return
        if len(frames) == 1 or self.role != "decompress":
            for frame in frames:
                self.consume(*frame)
            return
        if len(frames) > 16:
            raise ValueError("decompression batch exceeds the supported capacity of 16")
        # Reject the whole group before queuing reads or changing scan counters.
        for index, (_, nbytes, _, kind, projection, _) in enumerate(frames):
            received = self.received + index
            expected_kind = received if received < 2 else 2
            expected_projection = max(0, received - 2)
            if kind != expected_kind or projection != expected_projection:
                raise ValueError("missing or out-of-order compressed scan frame")
            if not 0 < nbytes <= self.compressed.nbytes:
                raise ValueError("compressed length outside the declared capacity")
            if received >= self.scan["angles"] + 2:
                raise ValueError("compressed batch extends beyond the scan")
        with self.stream:
            if not hasattr(self, "compressed_batch"):
                stride = ((self.compressed.nbytes + 255) // 256) * 256
                self.compressed_batch = cp.empty((16, stride), dtype=cp.uint8)
                self.batch_decoding = {}
            count = len(frames)
            if count not in self.batch_decoding:
                self.batch_decoding[count] = self.codec.decompression_config(
                    self.codec.compression_config([self.detector_frame_bytes] * count))
            sources, outputs = [], []
            for index, (pointer, nbytes, output_pointer, _, _, _) in enumerate(frames):
                compressed = self.compressed_batch[index, :nbytes]
                cp.copyto(compressed, array_at(pointer, (nbytes,), cp.uint8, self.gpu))
                sources.append(nvcomp.as_array(compressed, cuda_stream=self.stream.ptr))
                outputs.append(array_at(output_pointer, (self.detector_frame_bytes,),
                                        cp.uint8, self.gpu))
            self.codec.decode(sources, out=outputs, decompression_config=self.batch_decoding[count])
            self.received += count

    def finish(self):
        if self.role == "decompress" and self.received != self.scan["angles"] + 2:
            raise ValueError("incomplete compressed scan")
        if self.role != "decompress" and self.projections != self.scan["angles"]:
            raise ValueError("incomplete scan")

    def drain(self):
        """Complete every owned stream before cancellation can destroy its buffers.

        A next-block prefetch can remain queued if C++ cannot acquire the next
        publisher slot. Cleanup cannot rely only on the borrowed compute stream.
        """
        if hasattr(self, "host_buffer"):
            self.host_buffer.finish_downloads()
        if hasattr(self, "transfers"):
            self.transfers.finish()
        self.stream.synchronize()
