"""Exercise retained scans with deferred transfers that detect premature slot reuse."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from host_buffering import BlockTransfers, HostScanBuffer, memory_plan
from reconstruction import configuration, postprocess, reference_reconstruction
from test_reconstruction import disk_scan


class DeferredCUDA:
    """Copies read their source when the stream executes, like asynchronous DMA."""

    def __init__(self):
        runtime = self
        self.streams = []
        self.allocations = []

        class Stream:
            def __init__(self, *args, **kwargs):
                self.work = []
                self.completed = 0
                runtime.streams.append(self)

            def __enter__(self): return self
            def __exit__(self, *_): pass

            def enqueue(self, operation): self.work.append(operation)

            def complete(self, end):
                while self.completed < end:
                    operation = self.work[self.completed]
                    self.completed += 1
                    operation()

            def synchronize(self): self.complete(len(self.work))

            def wait_event(self, event):
                stream, end = event.stream, event.end
                self.enqueue(lambda: stream.complete(end))

        class Event:
            def __init__(self, **kwargs): pass

            def record(self, stream):
                self.stream, self.end = stream, len(stream.work)

            def synchronize(self): self.stream.complete(self.end)

        class Array(np.ndarray):
            def get(self, *, out=None, stream, blocking=True):
                if out is None:
                    out = np.empty(self.shape, self.dtype)
                stream.enqueue(lambda: np.copyto(out, self))
                if blocking:
                    stream.synchronize()
                return out

            def set(self, value, *, stream):
                stream.enqueue(lambda: np.copyto(self, value))

        def pinned(size):
            allocation = bytearray(size)
            self.allocations.append(allocation)
            return allocation

        self.Array = Array
        self.cp = SimpleNamespace(dtype=np.dtype, float32=np.float32, copyto=np.copyto,
            empty=lambda shape, dtype: np.empty(shape, dtype).view(Array),
            cuda=SimpleNamespace(Device=lambda gpu: SimpleNamespace(use=lambda: None),
                                 ExternalStream=Stream, Stream=Stream, Event=Event,
                                 alloc_pinned_memory=pinned))

    def load_processor(self):
        spec = importlib.util.spec_from_file_location("host_processor_test",
            Path(__file__).with_name("processors.py"))
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", cupy=self.cp, nvidia=SimpleNamespace(nvcomp=None)):
            spec.loader.exec_module(module)
        return module


def scan_configuration(rows=5, angles=3, columns=4, options=None):
    return dict(rows=rows, angles=angles, columns=columns,
                theta=np.linspace(0, np.pi, angles, endpoint=False).tolist(),
                reconstruction=options or configuration("sirt", slices_per_block=2),
                buffering=dict(sinogram_memory="host", host_budget_bytes=1024**2,
                               pinned_budget_bytes=1024**2))


class BudgetTests(unittest.TestCase):
    def test_budgets_cover_retained_scan_and_backend_specific_staging(self):
        scan = scan_configuration()
        for algorithm in ("sirt", "fbp", "gridrec"):
            options = configuration(algorithm, slices_per_block=2)
            plan = memory_plan(scan, options)
            # Five 3x4 sinograms; two 5x4 frames and two backend transfer blocks.
            required_pinned = 416 if algorithm == "gridrec" else 352
            self.assertEqual(plan["retained_host_bytes"], 240)
            self.assertEqual(plan["pinned_staging_bytes"], required_pinned)
            for key, required in (("host_budget_bytes", 240),
                                  ("pinned_budget_bytes", required_pinned)):
                with self.subTest(algorithm=algorithm, key=key):
                    scan["buffering"][key] = required - 1
                    with self.assertRaisesRegex(ValueError, key):
                        memory_plan(scan, options)
                    scan["buffering"][key] = required
                    memory_plan(scan, options)
                    scan["buffering"][key] = 1024**2

    def test_invalid_budgets_and_dimensions_fail_before_allocations(self):
        for key in ("host_budget_bytes", "pinned_budget_bytes"):
            for value in (0, -1, True, 1.5):
                scan = scan_configuration()
                scan["buffering"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    memory_plan(scan, scan["reconstruction"])
        for key in ("rows", "angles", "columns"):
            scan = scan_configuration()
            scan[key] = 0
            with self.subTest(key=key), self.assertRaises(ValueError):
                memory_plan(scan, scan["reconstruction"])


class OwnershipTests(unittest.TestCase):
    def test_scan_reset_drains_a_partial_scan_before_reusing_frames(self):
        runtime = DeferredCUDA()
        stream = runtime.cp.cuda.Stream()
        buffer = HostScanBuffer(scan_configuration(), runtime.cp, stream)
        buffer.append(np.full((5, 4), 1, np.float32).view(runtime.Array), 0)
        buffer.begin_scan()
        self.assertEqual(buffer.received, 0)
        self.assertEqual(buffer.pending, [None, None])
        np.testing.assert_array_equal(buffer.data[:, 0], 1)
        buffer.append(np.full((5, 4), 2, np.float32).view(runtime.Array), 0)
        buffer.finish_downloads()
        np.testing.assert_array_equal(buffer.data[:, 0], 2)

    def test_frame_slots_wait_before_packing_and_reuse_across_scans(self):
        runtime = DeferredCUDA()
        stream = runtime.cp.cuda.Stream()
        scan = scan_configuration(angles=5)
        buffer = HostScanBuffer(scan, runtime.cp, stream)
        staging = [id(frame) for frame in buffer.frames]
        retained = id(buffer.data)
        for scale in (1, 7):
            buffer.begin_scan()
            for projection in range(5):
                source = np.full((5, 4), (projection + 1) * scale, np.float32).view(runtime.Array)
                buffer.append(source, projection)
                # Model transport reuse ordered AFTER the queued read of its borrowed frame.
                stream.enqueue(lambda source=source: source.fill(-999))
            buffer.finish_downloads()
            for projection in range(5):
                np.testing.assert_array_equal(buffer.data[:, projection], (projection + 1) * scale)
        self.assertEqual([id(frame) for frame in buffer.frames], staging)
        self.assertEqual(id(buffer.data), retained)
        with self.assertRaisesRegex(ValueError, "in order"):
            buffer.append(source, 5)

    def test_prefetch_waits_for_last_consumer_before_overwriting_a_slot(self):
        runtime = DeferredCUDA()
        compute = runtime.cp.cuda.Stream()
        scan = scan_configuration()
        transfers = BlockTransfers(scan, memory_plan(scan, scan["reconstruction"]),
                                   runtime.cp, compute, cpu=False)
        results = []
        for index, rows in enumerate((2, 2, 1)):
            source = np.full((rows, 3, 4), index + 1, np.float32)
            slot = index % 2
            transfers.prefetch(source, slot)
            device = transfers.input(slot, rows)
            compute.enqueue(lambda device=device: results.append(device.copy()))
            transfers.consumed(slot)
        transfers.finish()
        for index, result in enumerate(results):
            np.testing.assert_array_equal(result, index + 1)
        self.assertEqual([result.shape[0] for result in results], [2, 2, 1])

    def test_cpu_output_slots_wait_for_upload_and_finish_drains_unmarked_work(self):
        runtime = DeferredCUDA()
        compute = runtime.cp.cuda.Stream()
        scan = scan_configuration(options=configuration("gridrec", slices_per_block=2))
        transfers = BlockTransfers(scan, memory_plan(scan, scan["reconstruction"]),
                                   runtime.cp, compute, cpu=True)
        output = np.full((5, 4, 4), np.nan, np.float32).view(runtime.Array)
        for index, (start, end) in enumerate(((0, 2), (2, 4), (4, 5))):
            result = np.full((end - start, 4, 4), index + 1, np.float32)
            transfers.upload_result(result, output[start:end], index % 2)
            result.fill(-999)  # TomoPy's returned storage can be released immediately.
        transfers.finish()
        np.testing.assert_array_equal(output[:, 0, 0], [1, 1, 2, 2, 3])
        compute.enqueue(lambda: output.fill(9))
        transfers.finish()  # Error cleanup cannot depend only on consumed() events.
        np.testing.assert_array_equal(output, 9)


class HostProcessorTests(unittest.TestCase):
    def test_failed_transfer_allocation_leaves_settings_unchanged_and_can_be_retried(self):
        runtime = DeferredCUDA()
        module = runtime.load_processor()
        processor = module.Processor("reconstruct", scan_configuration(), 0, 0, 40)
        previous = processor.reconstruction.copy()
        old_plan = processor.buffer_plan.copy()
        options = configuration("gridrec", slices_per_block=3)
        with patch.object(module, "BlockTransfers", side_effect=MemoryError("pinned allocation failed")):
            with self.assertRaisesRegex(MemoryError, "pinned allocation failed"):
                processor.configure_reconstruction(options)
        self.assertEqual(processor.reconstruction, previous)
        self.assertEqual(processor.buffer_plan, old_plan)
        self.assertIsNone(processor.transfer_layout)
        self.assertFalse(hasattr(processor, "transfers"))
        processor.configure_reconstruction(options)
        self.assertEqual(processor.reconstruction, options)
        self.assertTrue(processor.transfers.cpu)
        self.assertEqual(processor.transfer_layout, (3, True))

    def consume_scan(self, module, processor, runtime, source, output):
        for projection, angle in enumerate(processor.theta):
            frame = np.ascontiguousarray(source[:, projection]).view(runtime.Array)
            with patch.object(module, "array_at", side_effect=lambda pointer, *args: frame if pointer == 1 else output):
                processor.consume(1, frame.nbytes, 2 if projection + 1 == len(processor.theta) else 0,
                                  2, projection, angle)
        processor.finish()

    def test_actual_gridrec_matches_gpu_retention_with_gaussian_seams_and_repeated_scans(self):
        runtime = DeferredCUDA()
        module = runtime.load_processor()
        source, theta, _ = disk_scan(24)
        source = np.concatenate([source * np.float32(scale) for scale in (1, .2, .8, 0, .6)])
        options = configuration("gridrec", "hann", threads=2, slices_per_block=2,
                                gaussian_fwhm=2, scale_factor=1.3)
        scan = scan_configuration(angles=len(theta), columns=24, options=options)
        scan["theta"] = theta.tolist()
        with patch.object(module, "postprocess", side_effect=lambda volume, settings, **kw: postprocess(volume, settings)):
            host = module.Processor("reconstruct", scan, 0, 0, 40)
            gpu = module.Processor("reconstruct", dict(scan, buffering={"sinogram_memory": "gpu"}), 0, 0, 40)
            allocations = len(runtime.allocations)
            for scale in (1, .5):
                outputs = []
                for processor in (host, gpu):
                    processor.begin_scan()
                    processor.configure_reconstruction(options)
                    output = np.full((5, 24, 24), np.nan, np.float32).view(runtime.Array)
                    self.consume_scan(module, processor, runtime, source * scale, output)
                    outputs.append(output)
                expected = reference_reconstruction(source * scale, theta, 0, options)
                np.testing.assert_allclose(outputs[0], outputs[1], rtol=3e-5, atol=2e-6)
                np.testing.assert_allclose(outputs[0], expected, rtol=3e-5, atol=2e-6)
                np.testing.assert_array_equal(host.sinogram, source * scale)
            self.assertEqual(len(runtime.allocations), allocations)

    def test_backend_and_block_changes_rebuild_staging_and_reject_over_budget_settings(self):
        runtime = DeferredCUDA()
        module = runtime.load_processor()
        scan = scan_configuration()
        source = np.arange(60, dtype=np.float32).reshape(5, 3, 4)
        processor = module.Processor("reconstruct", scan, 0, 0, 40)

        def reconstruct(sinogram, output):
            processor.stream.enqueue(lambda: np.copyto(output,
                np.broadcast_to(sinogram.sum(axis=1)[:, None, :], output.shape)))

        with patch.object(processor, "_gpu_reconstruct_block", side_effect=reconstruct), \
             patch.object(module, "gridrec", side_effect=lambda sino, *args, **kw:
                 np.broadcast_to(sino.sum(axis=1)[:, None, :], (len(sino), 4, 4)).copy()), \
             patch.object(module, "postprocess", side_effect=lambda output, settings, **kw: postprocess(output, settings)):
            for method, size in (("sirt", 2), ("sirt", 2), ("fbp", 3), ("gridrec", 2)):
                processor.begin_scan()
                old_transfers = processor.transfers
                processor.configure_reconstruction(configuration(method, slices_per_block=size))
                if method == "sirt":
                    self.assertIs(processor.transfers, old_transfers)
                output = np.full((5, 4, 4), np.nan, np.float32).view(runtime.Array)
                self.consume_scan(module, processor, runtime, source, output)
                np.testing.assert_array_equal(output, np.broadcast_to(source.sum(axis=1)[:, None, :], output.shape))
            old_options, old_transfers = processor.reconstruction.copy(), processor.transfers
            scan["buffering"]["pinned_budget_bytes"] = 416
            with self.assertRaisesRegex(ValueError, "pinned_budget_bytes"):
                processor.configure_reconstruction(configuration("gridrec", slices_per_block=3))
            self.assertEqual(processor.reconstruction, old_options)
            self.assertIs(processor.transfers, old_transfers)

    def test_kernel_failure_drains_transfer_and_compute_work(self):
        runtime = DeferredCUDA()
        module = runtime.load_processor()
        scan = scan_configuration()
        processor = module.Processor("reconstruct", scan, 0, 0, 40)
        completed = []

        def fail(*args):
            processor.stream.enqueue(lambda: completed.append("last read finished"))
            raise RuntimeError("kernel launch failed")

        with patch.object(processor, "_gpu_reconstruct_block", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "kernel launch failed"):
                processor.reconstruct(np.empty((5, 4, 4), np.float32).view(runtime.Array))
        self.assertEqual(completed, ["last read finished"])
        for stream in runtime.streams:
            self.assertEqual(stream.completed, len(stream.work))


@unittest.skipUnless(os.environ.get("TOMOGRAPHY_TEST_GPU") == "1", "set TOMOGRAPHY_TEST_GPU=1")
class HostGPUIntegrationTests(unittest.TestCase):
    def test_host_retention_matches_existing_gpu_path_for_each_backend(self):
        import cupy as cp
        from processors import Processor

        source, theta, _ = disk_scan(24)
        source = np.concatenate([source * np.float32(scale) for scale in (1, .2, .8, 0, .6)])
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            frames = [cp.asarray(np.ascontiguousarray(source[:, index])) for index in range(len(theta))]
            for method in ("sirt", "fbp", "gridrec"):
                options = configuration(method, iterations=4, slices_per_block=2,
                                        gaussian_fwhm=1.5, scale_factor=.8)
                outputs = []
                for mode in ("gpu", "host"):
                    scan = scan_configuration(angles=len(theta), columns=24, options=options)
                    scan["theta"] = theta.tolist()
                    scan["buffering"]["sinogram_memory"] = mode
                    processor = Processor("reconstruct", scan, 0, stream.ptr, 4)
                    output = cp.empty((5, 24, 24), cp.float32)
                    for scale in (1, .5):
                        processor.begin_scan()
                        for index, (frame, angle) in enumerate(zip(frames, theta)):
                            scaled = frame * scale
                            processor.consume(scaled.data.ptr, scaled.nbytes,
                                output.data.ptr if index + 1 == len(theta) else 0, 2, index, angle)
                        processor.finish()
                        stream.synchronize()
                        expected = reference_reconstruction(source * scale, theta, 0, options)
                        np.testing.assert_allclose(output.get(), expected, rtol=3e-4, atol=2e-6)
                    outputs.append(output.get())
                np.testing.assert_allclose(outputs[0], outputs[1], rtol=3e-4, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
