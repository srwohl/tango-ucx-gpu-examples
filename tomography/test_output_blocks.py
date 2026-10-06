"""Bounded publisher views, retained host assembly, and complete-only previews."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from host_buffering import memory_plan
from output_blocks import LatestCompleted, OutputCollector
from reconstruction import configuration, postprocess, reference_reconstruction, slice_blocks
from test_host_buffering import DeferredCUDA, scan_configuration
from test_reconstruction import disk_scan


def block_scan(options=None):
    scan = scan_configuration(options=options)
    scan.update(scan_id=17, calibration_id=31)
    scan["buffering"].update(output_mode="blocks", output_block_rows=2,
                             output_host_budget_bytes=1024**2)
    return scan


def record(index, start, count, scan_id=17, calibration_id=31, revision=0):
    return dict(index=index, scan_id=scan_id, calibration_id=calibration_id,
                settings_revision=revision, slice_start=start, slice_count=count)


class CollectorTests(unittest.TestCase):
    def test_out_of_order_ranges_tail_and_ring_reuse_preserve_global_smoothing(self):
        options = configuration("fbp", slices_per_block=2, gaussian_fwhm=2, scale_factor=1.7)
        scan = block_scan(options)
        settings = lambda identity: dict(scan_id=identity, revision=0, options=options)
        collector = OutputCollector(scan, settings)
        source = np.arange(80, dtype=np.float32).reshape(5, 4, 4)
        slot = np.full((2, 4, 4), -1234, np.float32)
        for index, (start, count) in enumerate(((2, 2), (0, 2), (4, 1))):
            slot[:count] = source[start:start + count]
            result = collector.add(slot, record(index, start, count))
            slot.fill(-999)  # The transport can immediately reuse its receive slot.
            if index < 2:
                self.assertIsNone(result)
        np.testing.assert_allclose(result[0], postprocess(source.copy(), options))
        collector.finish()
        self.assertEqual(collector.completed, 1)

    def test_missing_duplicate_overlap_and_identity_changes_are_rejected(self):
        scan = block_scan()
        settings = lambda identity: dict(scan_id=identity, revision=0, options=scan["reconstruction"])
        slot = np.ones((2, 4, 4), np.float32)
        for wrong, text in ((record(2, 2, 2), "missing or out-of-order"),
                            (record(1, 1, 2), "overlapping"),
                            (record(1, 2, 2, scan_id=18), "changed"),
                            (record(1, 2, 2, revision=1), "changed"),
                            (record(1, 4, 2), "extent")):
            collector = OutputCollector(scan, settings)
            collector.add(slot, record(0, 0, 2))
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, text):
                collector.add(slot, wrong)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                collector.finish()
        with self.assertRaisesRegex(ValueError, "revision"):
            OutputCollector(scan, settings).add(slot, record(0, 0, 2, revision=1))
        with self.assertRaisesRegex(ValueError, "calibration"):
            OutputCollector(scan, settings).add(slot, record(0, 0, 2, calibration_id=30))

    def test_repeated_scan_settings_and_block_layout_transitions_reuse_host_volume(self):
        scan = block_scan()
        first = scan["reconstruction"]
        second = configuration("gridrec", slices_per_block=1, scale_factor=2)
        settings = lambda identity: dict(scan_id=identity, revision=identity - 17,
                                         options=first if identity == 17 else second)
        collector = OutputCollector(scan, settings)
        allocation = id(collector.data)
        index = 0
        for cycle, size in ((0, 2), (1, 1)):
            for block in slice_blocks(5, size):
                count = block.stop - block.start
                result = collector.add(np.ones((2, 4, 4), np.float32),
                    record(index, block.start, count, 17 + cycle, 31 + cycle, cycle))
                index += 1
            self.assertEqual(result[3]["revision"], cycle)
            np.testing.assert_array_equal(result[0], 1 + cycle)
            self.assertEqual(id(collector.data), allocation)
        collector.finish()

    def test_snapshot_mailbox_copies_complete_volumes_and_replaces_pending(self):
        mailbox = LatestCompleted()
        volume = np.ones((5, 4, 4), np.float32)
        mailbox.put((volume, 17, 0, {}), {})
        volume.fill(2)
        first = mailbox.take()
        mailbox.put((volume, 18, 1, {}), {})
        volume.fill(3)
        mailbox.put((volume, 19, 2, {}), {})
        volume.fill(-999)
        np.testing.assert_array_equal(first[0], 1)
        latest = mailbox.take()
        self.assertEqual(latest[1:3], (19, 2))
        np.testing.assert_array_equal(latest[0], 3)
        mailbox.close()
        self.assertIsNone(mailbox.take())


class BufferBudgetTests(unittest.TestCase):
    def test_logical_volume_can_exceed_frame_limit_and_live_capacity_stays_fixed(self):
        scan = dict(rows=2048, angles=2048, columns=2048, buffering=dict(
            sinogram_memory="host", output_mode="blocks", output_block_rows=16,
            host_budget_bytes=64 * 1024**3, pinned_budget_bytes=1024**3,
            output_host_budget_bytes=64 * 1024**3))
        plan = memory_plan(scan, configuration("sirt", slices_per_block=16))
        self.assertEqual(plan["gpu_output_bytes"], 256 * 1024**2)
        self.assertEqual(plan["host_output_bytes"], 32 * 1024**3)
        self.assertEqual(plan["gpu_sinogram_bytes"], 0)
        smaller = memory_plan(scan, configuration("sirt", slices_per_block=8))
        self.assertEqual(smaller["gpu_output_bytes"], plan["gpu_output_bytes"])
        for size in (0, 17):
            with self.assertRaisesRegex(ValueError, "capacity"):
                memory_plan(scan, configuration("sirt", slices_per_block=size))
        scan["buffering"]["output_host_budget_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "output_host_budget"):
            memory_plan(scan, configuration("sirt", slices_per_block=8))


class ProcessorOutputTests(unittest.TestCase):
    def test_cancellation_between_publications_drains_pending_next_block_prefetch(self):
        runtime = DeferredCUDA()
        module = runtime.load_processor()
        processor = module.Processor("reconstruct", block_scan(), 0, 0, 40)
        source = np.arange(60, dtype=np.float32).reshape(5, 3, 4)
        self.consume(module, processor, runtime, source)
        output = np.empty((2, 4, 4), np.float32).view(runtime.Array)
        def compute(sinogram, volume):
            processor.stream.enqueue(lambda: np.copyto(volume,
                np.broadcast_to(sinogram.sum(axis=1)[:, None], volume.shape)))
        with patch.object(module, "array_at", return_value=output), \
             patch.object(processor, "_gpu_reconstruct_block", side_effect=compute):
            processor.reconstruct_block(1, 0, 2)
        # Model cancellation while C++ is waiting for the next publisher slot.
        self.assertLess(processor.transfers.transfer_stream.completed,
                        len(processor.transfers.transfer_stream.work))
        processor.drain()
        for stream in runtime.streams:
            self.assertEqual(stream.completed, len(stream.work))
        np.testing.assert_array_equal(output, np.broadcast_to(source[:2].sum(axis=1)[:, None], output.shape))

    def consume(self, module, processor, runtime, source):
        for projection, angle in enumerate(processor.theta):
            frame = np.ascontiguousarray(source[:, projection]).view(runtime.Array)
            with patch.object(module, "array_at", return_value=frame):
                # No publisher pointer is available or needed while collecting a scan.
                processor.consume(1, frame.nbytes, 0, 2, projection, angle)

    def collect(self, module, processor, runtime, collector, index=0):
        output = np.full((2, processor.scan["columns"], processor.scan["columns"]), np.nan,
                         np.float32).view(runtime.Array)
        for block in slice_blocks(processor.scan["rows"], processor.reconstruction["slices_per_block"]):
            count = block.stop - block.start
            output.fill(np.nan)
            with patch.object(module, "array_at", return_value=output), \
                 patch.object(module, "postprocess", side_effect=AssertionError("GPU postprocess forbidden")):
                processor.reconstruct_block(1, block.start, count)
            processor.stream.synchronize()  # Model the publisher completion before host access.
            np.testing.assert_array_equal(output[count:], 0)
            result = collector.add(output, record(index, block.start, count))
            output.fill(-999)
            index += 1
        return result, index

    def test_gpu_input_prefetch_tail_direct_outputs_and_error_cleanup(self):
        runtime = DeferredCUDA()
        module = runtime.load_processor()
        scan = block_scan()
        processor = module.Processor("reconstruct", scan, 0, 0, 40)
        source = np.arange(60, dtype=np.float32).reshape(5, 3, 4)
        collector = OutputCollector(scan, lambda identity: dict(scan_id=identity, revision=0,
                                                               options=scan["reconstruction"]))
        self.consume(module, processor, runtime, source)

        def reconstruct(sinogram, volume):
            processor.stream.enqueue(lambda: np.copyto(volume,
                np.broadcast_to(sinogram.sum(axis=1)[:, None, :], volume.shape)))

        with patch.object(processor, "_gpu_reconstruct_block", side_effect=reconstruct):
            result, _ = self.collect(module, processor, runtime, collector)
        np.testing.assert_array_equal(result[0], np.broadcast_to(source.sum(axis=1)[:, None], (5, 4, 4)))
        self.assertEqual(processor.transfers.used, [True, True])
        processor.begin_scan()
        self.consume(module, processor, runtime, source)
        with patch.object(module, "array_at", return_value=np.empty((2, 4, 4), np.float32).view(runtime.Array)), \
             patch.object(processor, "_gpu_reconstruct_block", side_effect=RuntimeError("kernel failed")):
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                processor.reconstruct_block(1, 0, 2)
        for stream in runtime.streams:
            self.assertEqual(stream.completed, len(stream.work))

    def test_actual_cpu_gridrec_blocks_match_reference_with_global_smoothing(self):
        runtime = DeferredCUDA()
        module = runtime.load_processor()
        source, theta, _ = disk_scan(24)
        source = np.concatenate([source * scale for scale in (1, .2, .8, 0, .6)])
        options = configuration("gridrec", "hann", threads=2, slices_per_block=2,
                                gaussian_fwhm=2, scale_factor=1.3)
        scan = block_scan(options)
        scan.update(columns=24, angles=len(theta), theta=theta.tolist())
        processor = module.Processor("reconstruct", scan, 0, 0, 40)
        collector = OutputCollector(scan, lambda identity: dict(scan_id=identity, revision=0, options=options))
        self.consume(module, processor, runtime, source)
        result, _ = self.collect(module, processor, runtime, collector)
        expected = reference_reconstruction(source, theta, 0, options)
        np.testing.assert_allclose(result[0], expected, rtol=3e-5, atol=2e-6)
        # The independent reference can also use bounded backend allocations.
        np.testing.assert_allclose(reference_reconstruction(source, theta, 0, options, block_rows=2),
                                   expected, rtol=3e-5, atol=2e-6)


@unittest.skipUnless(os.environ.get("TOMOGRAPHY_TEST_GPU") == "1", "set TOMOGRAPHY_TEST_GPU=1")
class CUDAOutputTests(unittest.TestCase):
    def test_actual_sirt_fbp_and_gridrec_publish_only_block_sized_gpu_outputs(self):
        import cupy as cp
        from processors import Processor

        source, theta, _ = disk_scan(24)
        source = np.concatenate([source * scale for scale in (1, .2, .8, 0, .6)])
        stream = cp.cuda.Stream(non_blocking=True)
        for method in ("sirt", "fbp", "gridrec"):
            options = configuration(method, iterations=4, slices_per_block=2,
                                    gaussian_fwhm=1.5, scale_factor=.8)
            scan = block_scan(options)
            scan.update(columns=24, angles=len(theta), theta=theta.tolist())
            processor = Processor("reconstruct", scan, 0, stream.ptr, 4)
            collector = OutputCollector(scan, lambda identity: dict(scan_id=identity, revision=0, options=options))
            with stream:
                frame = cp.empty((5, 24), cp.float32)
                slot = cp.empty((2, 24, 24), cp.float32)
                for projection, angle in enumerate(theta):
                    frame.set(np.ascontiguousarray(source[:, projection]), stream=stream)
                    processor.consume(frame.data.ptr, frame.nbytes, 0, 2, projection, angle)
                for index, block in enumerate(slice_blocks(5, 2)):
                    count = block.stop - block.start
                    processor.reconstruct_block(slot.data.ptr, block.start, count)
                    stream.synchronize()
                    output = slot.get()
                    np.testing.assert_array_equal(output[count:], 0)
                    result = collector.add(output, record(index, block.start, count))
                processor.finish()
            expected = reference_reconstruction(source, theta, 0, options, block_rows=2)
            np.testing.assert_allclose(result[0], expected, rtol=3e-4, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
