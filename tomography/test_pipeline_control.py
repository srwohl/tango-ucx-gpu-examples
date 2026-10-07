"""CPU checks for pipeline settings validation and atomic restart requests."""
import json
from pathlib import Path
import tempfile
import threading
import unittest

from pipeline_control import (AUTO_LINK_BYTES, MAX_BUFFERED_FRAMES, MAX_GPU_RING_FRAMES, MAX_RECONSTRUCTORS,
                              PipelineConflict, PipelineControl, atomic_json, link_budget, scan_ring_budget,
                              validate_options)


OPTIONS = dict(pixels=32, slices=16, angles=90, network="auto", net_devices=None, transport_batch=4, processing_batch=4,
               processing_mode="scalar", sinogram_memory="gpu", receive_budget_mib=4.0,
               host_buffer_mib=64, pinned_buffer_mib=16, output_mode="volume",
               output_host_mib=64, gpu=0, decompress_gpu=0, correct_gpu=0,
               reconstruct_gpu=0, scan_period=0.0)


def running_state():
    return dict(requested=dict(options=OPTIONS.copy(), revision=0),
                active=dict(options=OPTIONS.copy(), revision=0), phase="running", run_id=0,
                run_output="run-0", error=None, gpu_count=2)


class ValidationTests(unittest.TestCase):
    def test_all_settings_and_partial_updates(self):
        self.assertEqual(validate_options(OPTIONS, 2), OPTIONS)
        self.assertEqual(validate_options(dict(decompress_gpu=1, scan_period=0), 2),
                         dict(decompress_gpu=1, scan_period=0))

    def test_unknown_types_ranges_and_nonfinite_values(self):
        invalid = [None, [], {}, {"surprise": 1}, {"network": "ethernet"},
                   {"network": ["auto"]}, {"processing_mode": "parallel"},
                   {"sinogram_memory": "cuda"}, {"output_mode": "latest"},
                   {"net_devices": 1}, {"net_devices": ""}, {"net_devices": "lo\n"}]
        for key in ("transport_batch", "processing_batch"):
            invalid += [{key: value} for value in (0, 17, 1.5, True)]
        for key in ("gpu", "decompress_gpu", "correct_gpu", "reconstruct_gpu"):
            invalid += [{key: value} for value in (-1, 2, 0.0, True)]
        for key in ("host_buffer_mib", "pinned_buffer_mib", "output_host_mib", "pixels", "slices", "angles"):
            invalid += [{key: value} for value in (0, -1, 1.5, True)]
        for key in ("receive_budget_mib", "scan_period"):
            invalid += [{key: value} for value in (-1, True, "1", float("nan"), float("inf"), 10**400)]
        invalid.append({"receive_budget_mib": 0})
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_options(options, 2)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.state = running_state()
        self.control = PipelineControl(self.root)
        atomic_json(self.control.state_path, self.state)

    def tearDown(self):
        self.directory.cleanup()

    def test_request_is_atomic_and_does_not_mark_options_active(self):
        reply = self.control.configure(dict(network="tcp", processing_mode="batched"))
        request = json.loads(self.control.request_path.read_text())
        self.assertEqual(request["revision"], 1)
        self.assertEqual(request["options"], dict(OPTIONS, network="tcp", processing_mode="batched"))
        self.assertEqual(reply["requested"], request)
        self.assertEqual(reply["active"], self.state["active"])
        self.assertEqual(json.loads(self.control.state_path.read_text()), self.state)
        self.assertEqual(self.control.settings()["requested"], request)
        self.assertFalse(list(self.root.glob("*.tmp")))
        with self.assertRaises(PipelineConflict):
            self.control.configure(dict(transport_batch=16))

    def test_only_one_concurrent_http_request_is_accepted(self):
        barrier = threading.Barrier(2)
        accepted, rejected = [], []
        def submit(batch):
            barrier.wait()
            try:
                accepted.append(self.control.configure(dict(transport_batch=batch)))
            except PipelineConflict as error:
                rejected.append(str(error))
        threads = [threading.Thread(target=submit, args=(batch,)) for batch in (1, 16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual((len(accepted), len(rejected)), (1, 1))
        self.assertEqual(self.control.settings()["requested"]["revision"], 1)

    def test_supervisor_can_apply_then_accept_the_next_revision(self):
        request = self.control.configure(dict(scan_period=.5))["requested"]
        self.state.update(active=request, requested=request, run_id=1)
        atomic_json(self.control.state_path, self.state)
        self.control.request_path.unlink()
        reply = self.control.configure(dict(transport_batch=8))
        self.assertEqual(reply["requested"]["revision"], 2)
        self.assertEqual(reply["requested"]["options"]["scan_period"], .5)

    def test_finished_restarting_failed_and_pending_states_reject_writes(self):
        for phase in ("finished", "restarting", "failed"):
            atomic_json(self.control.state_path, dict(self.state, phase=phase))
            with self.subTest(phase=phase), self.assertRaises(PipelineConflict):
                self.control.configure(dict(network="tcp"))
        pending = dict(self.state, requested=dict(options=OPTIONS, revision=1))
        atomic_json(self.control.state_path, pending)
        with self.assertRaises(PipelineConflict):
            self.control.configure(dict(network="tcp"))
        self.assertFalse(self.control.request_path.exists())

    def test_fixed_hdf5_geometry_and_batched_processing_constraint(self):
        atomic_json(self.control.state_path, dict(self.state, fixed_geometry=True))
        with self.assertRaisesRegex(ValueError, "geometry is fixed"):
            self.control.configure(dict(pixels=64))
        self.assertFalse(self.control.request_path.exists())
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            self.control.configure(dict(processing_mode="batched", processing_batch=16))
        self.assertFalse(self.control.request_path.exists())
        reply = self.control.configure(dict(pixels=32, transport_batch=16,
                                             processing_mode="batched", processing_batch=16))
        self.assertEqual(reply["requested"]["options"]["pixels"], 32)

    def test_invalid_request_does_not_write_a_file(self):
        with self.assertRaises(ValueError):
            self.control.configure(dict(gpu=2))
        self.assertFalse(self.control.request_path.exists())

    def test_stop_cancels_pending_restart_and_is_idempotent(self):
        self.control.configure(dict(transport_batch=16))
        reply = self.control.stop()
        self.assertEqual(reply["phase"], "stopping")
        self.assertEqual(reply["requested"], self.state["active"])
        self.assertTrue(self.control.stop_path.exists())
        self.assertFalse(self.control.request_path.exists())
        self.assertEqual(self.control.stop()["phase"], "stopping")
        with self.assertRaises(PipelineConflict):
            self.control.configure(dict(network="tcp"))
        atomic_json(self.control.state_path, dict(self.state, phase="finished"))
        self.assertEqual(self.control.settings()["phase"], "finished")

    def test_buffer_recommendation_uses_image_and_does_not_change_running_settings(self):
        workload = dict(detector_element="f32", reconstruction=dict(algorithm="fbp", slices_per_block=4))
        atomic_json(self.control.state_path, dict(self.state, workload=workload))
        result = self.control.recommend(dict(transport_batch=16, processing_batch=16,
                                             processing_mode="batched"))
        info = result["information"]
        self.assertEqual(info["detector_shape"], [16, 32])
        self.assertEqual(info["detector_frame_bytes"], 16 * 32 * 4)
        self.assertEqual(info["sinogram_bytes"], 16 * 90 * 32 * 4)
        self.assertGreaterEqual(result["options"]["host_buffer_mib"] * 1024**2, info["sinogram_bytes"])
        self.assertGreaterEqual(result["options"]["pinned_buffer_mib"] * 1024**2, info["pinned_staging_bytes"])
        self.assertGreaterEqual(result["options"]["output_host_mib"] * 1024**2, info["volume_bytes"] * 3)
        self.assertEqual((info["buffered_frames"], info["scan_frames"]), (92, 92))
        self.assertGreaterEqual(result["options"]["receive_budget_mib"] * 1024**2,
                                92 * (info["corrected_frame_bytes"] + 4096))
        self.assertEqual(self.control.settings()["active"], self.state["active"])
        self.assertFalse(self.control.request_path.exists())

    def test_link_budget_holds_a_scan_within_transport_and_memory_limits(self):
        budget, frames = link_budget(128, 256, 720, 16)
        self.assertEqual(frames, 722)
        self.assertGreaterEqual(budget, 722 * 128 * 256 * 4)
        self.assertEqual(link_budget(8, 64, 3000, 1)[1], MAX_BUFFERED_FRAMES)
        budget, frames = link_budget(2048, 2048, 1800, 4)
        self.assertEqual(frames, AUTO_LINK_BYTES // (2048 * 2048 * 4 + 4096))
        self.assertLessEqual(budget, AUTO_LINK_BYTES + (256 << 10))
        self.assertEqual(link_budget(4096, 4096, 1800, 16)[1], 32)

    def test_several_reconstructors_size_the_receive_ring_for_a_whole_scan(self):
        self.assertEqual(validate_options(dict(reconstructors=MAX_RECONSTRUCTORS))["reconstructors"], 8)
        for value in (0, 9, 2.0, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "reconstructors"):
                validate_options(dict(reconstructors=value))
        self.assertIsNone(scan_ring_budget(8, 64, MAX_GPU_RING_FRAMES + 1, 1))
        self.assertGreaterEqual(scan_ring_budget(128, 256, 720, 16), (720 + 16) * 128 * 256 * 4)
        # A 2048-pixel scan exceeds the automatic per-link limit; a pull set still needs all of it.
        options = dict(pixels=2048, slices=1024, angles=96, transport_batch=4, processing_batch=4)
        one = self.control.recommend(options)
        several = self.control.recommend(dict(options, reconstructors=2))
        self.assertLess(one["information"]["buffered_frames"], 98)
        self.assertEqual(several["information"]["buffered_frames"], 98)
        self.assertGreaterEqual(several["options"]["receive_budget_mib"] * 1024**2,
                                scan_ring_budget(1024, 2048, 96, 4))
        with self.assertRaisesRegex(ValueError, "4096 frames"):
            self.control.recommend(dict(options, angles=5000, reconstructors=2))

    def test_buffer_recommendation_keeps_hdf5_geometry(self):
        atomic_json(self.control.state_path, dict(self.state, fixed_geometry=True))
        with self.assertRaisesRegex(ValueError, "geometry is fixed"):
            self.control.recommend(dict(pixels=64))


if __name__ == "__main__":
    unittest.main()
