"""CPU-only history and HTTP checks; no Tango server or GPU required."""
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from pipeline_control import MAX_BATCH, PipelineControl, atomic_json
from test_pipeline_control import running_state
from reconstruction import configuration
from viewer import ReconstructionControl, VolumeHistory, display_voxels, make_handler

HEALTH = dict(skipped=0, transport="tcp")


class HistoryTests(unittest.TestCase):
    def test_sixteen_bit_display_preserves_sub_eight_bit_detail(self):
        source = np.array([[[.005, .005001, .005002]]], dtype=np.float32)
        original = source.copy()
        history = VolumeHistory()
        history.append(source, 1, 0, HEALTH)
        voxels, meta = history.get(1)
        self.assertEqual(voxels.dtype, np.dtype('<u2'))
        self.assertEqual(len(np.unique(voxels)), 3)
        self.assertEqual(meta["bytes"], source.size * 2)
        np.testing.assert_array_equal(source, original)

    def test_oversized_uint16_volume_does_not_evict_buffered_data(self):
        history = VolumeHistory(max_bytes=12)
        history.append(np.zeros((1, 2, 3)), 1, 0, HEALTH)
        with self.assertRaisesRegex(ValueError, "byte limit"):
            history.append(np.zeros((1, 2, 4)), 2, 1, HEALTH)
        self.assertEqual(history.snapshot()["version"], 1)
        self.assertIsNotNone(history.get(1))

    def test_custom_display_range_preserves_real_data_contrast(self):
        source = np.array([[[0, .02, .04, .08]]], dtype=np.float32)
        original = source.copy()
        history = VolumeHistory(display_max=.04)
        history.append(source, 1, 0, HEALTH)
        np.testing.assert_array_equal(history.get(1)[0], [[[0, 32767, 65535, 65535]]])
        np.testing.assert_array_equal(source, original)
        self.assertEqual(history.snapshot()["display_max"], .04)
        for maximum in (0, -1, np.nan, np.inf):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                VolumeHistory(display_max=maximum)

    def test_buffered_scans_keep_their_original_reconstruction_settings(self):
        history = VolumeHistory()
        first = dict(options=configuration("gridrec"), revision=0, scan_id=40)
        second = dict(options=configuration("sirt", iterations=10), revision=1, scan_id=41)
        history.append(np.zeros((1, 2, 3)), 40, 0, HEALTH, first)
        history.append(np.zeros((1, 2, 3)), 41, 1, HEALTH, second)
        self.assertEqual(history.get(1)[1]["reconstruction"], first)
        self.assertEqual(history.get(2)[1]["reconstruction"], second)

    def test_count_bound_and_stable_scan_identity(self):
        history = VolumeHistory(max_volumes=2)
        source = np.full((2, 3, 4), .005, dtype=np.float32)
        history.append(source, 100, 0, HEALTH)
        original, _ = history.get(1)
        source.fill(.01)
        history.append(source, 102, 2, HEALTH)
        history.append(source, 105, 5, HEALTH)
        state = history.snapshot()
        self.assertEqual([f["scan_id"] for f in state["volumes"]], [102, 105])
        self.assertEqual(state["skipped"], 3)
        self.assertEqual(state["buffered_bytes"], 96)
        self.assertIsNone(history.get(1))
        self.assertTrue(np.all(original == 32767))
        self.assertFalse(original.flags.writeable)
        self.assertEqual(history.get(2)[1]["scan_id"], 102)
        self.assertGreaterEqual(state["volumes"][1]["time_seconds"], state["volumes"][0]["time_seconds"])

    def test_byte_bound_with_changing_shapes(self):
        history = VolumeHistory(max_volumes=10, max_bytes=80)
        for i, shape in enumerate([(2, 3, 4), (2, 2, 4), (1, 3, 4)]):
            history.append(np.zeros(shape), i, i, HEALTH)
        self.assertEqual(history.snapshot()["buffered_bytes"], 56)
        self.assertEqual(history.snapshot()["buffered_volumes"], 2)
        with self.assertRaisesRegex(ValueError, "byte limit"):
            history.append(np.zeros((2, 5, 5)), 3, 3, HEALTH)
        self.assertEqual(history.snapshot()["received_volumes"], 3)

    def test_updates_within_a_scan_advance_by_projection_and_keep_their_planes(self):
        history = VolumeHistory()
        planes = [dict(origin=[0, 0, 0], u=[0, 1, 0], v=[0, 0, 1])] * 3
        history.append(np.zeros((3, 4, 4)), 5, 0, HEALTH, projection=95, planes=planes)
        history.append(np.zeros((3, 4, 4)), 6, 1, HEALTH, projection=15, planes=planes)
        history.append(np.zeros((3, 4, 4)), 6, 3, HEALTH, projection=47)
        for scan, projection in ((6, 47), (6, 31), (5, 95)):
            with self.subTest(scan=scan, projection=projection), self.assertRaises(ValueError):
                history.append(np.zeros((3, 4, 4)), scan, 4, HEALTH, projection=projection)
        state = history.snapshot()
        self.assertEqual((state["scan_id"], state["projection"], state["received_volumes"], state["skipped"]),
                         (6, 47, 3, 1))
        self.assertEqual([(frame["scan_id"], frame["projection"], frame.get("planes")) for frame in state["volumes"]],
                         [(5, 95, planes), (6, 15, planes), (6, 47, None)])
        history.append(np.zeros((3, 4, 4)), 7, 4, HEALTH, projection=15)

    def test_invalid_volume_or_repeated_scan_does_not_replace_history(self):
        history = VolumeHistory()
        history.append(np.zeros((1, 2, 3)), 1, 0, HEALTH)
        for volume, scan in [(np.zeros((1, 2, 3)), 1),
                             (np.full((1, 2, 3), np.nan), 2),
                             (np.zeros((2, 3)), 2)]:
            with self.assertRaises(ValueError):
                history.append(volume, scan, 1, HEALTH)
        self.assertEqual(history.snapshot()["version"], 1)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.history = VolumeHistory(max_volumes=2)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0),
            make_handler(self.history, Path(self.directory.name)))
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def get(self, path):
        return urllib.request.urlopen(self.url + path, timeout=3)

    def test_volume_format_shape_and_requested_version(self):
        source = np.linspace(-.01, .02, 24).reshape(2, 3, 4)
        self.history.append(source, 40, 0, HEALTH)
        self.history.append(np.zeros_like(source), 41, 1, HEALTH)
        with self.get("/api/volume?v=1") as response:
            self.assertEqual(response.headers["X-Volume-Shape"], "2,3,4")
            self.assertEqual(response.headers["X-Volume-Version"], "1")
            self.assertEqual(response.headers["X-Scan-Id"], "40")
            self.assertEqual(response.headers["X-Volume-Format"], "uint16-le-zyx")
            self.assertEqual(response.read(), display_voxels(source).tobytes())
        for axis in range(3):
            with self.get(f"/api/slice?v=1&axis={axis}&index=0") as response:
                with Image.open(BytesIO(response.read())) as image:
                    np.testing.assert_array_equal(np.asarray(image),
                        (np.take(display_voxels(source), 0, axis=axis).astype(float) / 257).astype(np.uint8))
        with self.get("/api/status") as response:
            state = json.load(response)["viewer"]
            self.assertEqual([f["version"] for f in state["volumes"]], [1, 2])

    def test_custom_display_range_is_used_for_volume_and_slices(self):
        self.history = VolumeHistory(max_volumes=2, display_max=.04)
        self.server.RequestHandlerClass = make_handler(self.history, Path(self.directory.name))
        source = np.linspace(0, .04, 24, dtype=np.float32).reshape(2, 3, 4)
        self.history.append(source, 1, 0, HEALTH)
        with self.get("/api/volume") as response:
            self.assertEqual(response.headers["X-Display-Range"], "0,0.04")
            self.assertEqual(response.read(), display_voxels(source, .04).tobytes())
        with self.get("/api/slice?axis=0&index=0") as response:
            with Image.open(BytesIO(response.read())) as image:
                np.testing.assert_array_equal(np.asarray(image), (display_voxels(source, .04)[0].astype(float) / 257).astype(np.uint8))
        with self.get("/api/slice?axis=0&index=0&window_max=0.02") as response:
            with Image.open(BytesIO(response.read())) as image:
                expected = np.clip(display_voxels(source, .04)[0].astype(float) * (2 / 257), 0, 255).astype(np.uint8)
                np.testing.assert_array_equal(np.asarray(image), expected)

    def test_empty_expired_and_invalid_requests(self):
        with self.get("/api/volume") as response:
            self.assertEqual(response.status, 204)
        for i in range(3):
            self.history.append(np.zeros((1, 2, 3)), i, i, HEALTH)
        for path, status in [("/api/volume?v=1", 410), ("/api/slice?v=1", 410),
                             ("/api/volume?v=bad", 400), ("/api/slice?axis=3", 400),
                             ("/api/slice?axis=0&index=1", 400)]:
            with self.assertRaises(urllib.error.HTTPError) as result:
                self.get(path)
            self.assertEqual(result.exception.code, status)
            result.exception.close()
        with self.get("/api/volume") as response:
            self.assertEqual(response.headers["X-Volume-Version"], "3")

    def test_slice_window_low_is_applied_after_sixteen_bit_quantization(self):
        source = np.array([[[.002, .004, .006, .008]]], dtype=np.float32)
        self.history.append(source, 1, 0, HEALTH)
        with self.get("/api/slice?axis=0&index=0&window_min=0.004&window_max=0.008") as response:
            with Image.open(BytesIO(response.read())) as image:
                values = display_voxels(source)[0].astype(float) * (.01 / 65535)
                expected = np.clip((values - .004) * (255 / .004), 0, 255).astype(np.uint8)
                np.testing.assert_array_equal(np.asarray(image), expected)
        for minimum in ("-1", "nan", "inf", "0.01", "0.02"):
            with self.subTest(minimum=minimum), self.assertRaises(urllib.error.HTTPError) as result:
                self.get(f"/api/slice?window_min={minimum}")
            self.assertEqual(result.exception.code, 400)
            result.exception.close()

    def test_interior_window_reveals_detail_on_each_axis_without_changing_volume(self):
        source = np.linspace(0, .01, 24, dtype=np.float32).reshape(2, 3, 4)
        self.history.append(source, 40, 0, HEALTH)
        original = display_voxels(source)
        for axis in range(3):
            with self.get(f"/api/slice?v=1&axis={axis}&index=0&window_max=0.004") as response:
                with Image.open(BytesIO(response.read())) as image:
                    expected = np.clip(np.take(original, 0, axis=axis).astype(float) * (2.5 / 257),
                                       0, 255).astype(np.uint8)
                    np.testing.assert_array_equal(np.asarray(image), expected)
        with self.get("/api/volume?v=1") as response:
            self.assertEqual(response.read(), original.tobytes())
        for window in ("0", "-0.004", "0.02", "nan", "inf", "bad"):
            with self.subTest(window=window), self.assertRaises(urllib.error.HTTPError) as result:
                self.get(f"/api/slice?window_max={window}")
            self.assertEqual(result.exception.code, 400)
            result.exception.close()


class ControlHTTPTests(unittest.TestCase):
    tearDown = HTTPTests.tearDown
    get = HTTPTests.get

    def setUp(self):
        HTTPTests.setUp(self)
        class Proxy:
            def __init__(self):
                self.calls = []
                self.failure = None
                self.state = dict(requested=dict(options=configuration("gridrec"), revision=0),
                                  active=None, detector_columns=64, finished=False)

            def command_inout(self, command, argument=None):
                self.calls.append((command, argument))
                if self.failure:
                    raise RuntimeError(self.failure)
                if command == "ConfigureReconstruction":
                    self.state["requested"] = dict(options=json.loads(argument), revision=1)
                return json.dumps(self.state)
        self.proxy = Proxy()
        self.server.RequestHandlerClass = make_handler(
            self.history, Path(self.directory.name), ReconstructionControl(self.proxy))

    def post(self, options, **headers):
        request = urllib.request.Request(self.url + "/api/reconstruction",
            data=json.dumps(options).encode(), headers={"Content-Type": "application/json", **headers})
        return urllib.request.urlopen(request, timeout=3)

    def test_read_and_apply_controls_use_tango_commands(self):
        with self.get("/api/reconstruction") as response:
            self.assertEqual(json.load(response)["requested"]["options"]["algorithm"], "gridrec")
        with self.post(dict(algorithm="sirt", iterations=12, min_constraint=None)) as response:
            state = json.load(response)
            self.assertEqual(state["requested"]["options"]["iterations"], 12)
            self.assertIsNone(state["requested"]["options"]["min_constraint"])
            self.assertIsNone(state["active"])
        with self.get("/api/status") as response:
            self.assertEqual(json.load(response)["reconstruction_control"], state)
        self.assertEqual([command for command, _ in self.proxy.calls],
                         ["GetReconstruction", "GetReconstruction", "ConfigureReconstruction", "GetReconstruction"])

    def test_slice_planes_are_validated_here_and_moved_on_the_device(self):
        planes = [dict(origin=[4, 0, 0], u=[0, 1, 0], v=[0, 0, 1]),
                  dict(origin=[-2.5, 30, 1], u=[.9, .2, 0], v=[0, 0, 1.5]),
                  dict(origin=[0, 0, 32], u=[1, 0, 0], v=[0, 1, 0])]
        send = lambda value: urllib.request.urlopen(urllib.request.Request(
            self.url + "/api/slices", data=json.dumps(value).encode(),
            headers={"Content-Type": "application/json"}), timeout=3)
        with send(planes) as response:
            self.assertEqual(json.load(response), self.proxy.state)
        command, argument = self.proxy.calls[-1]
        self.assertEqual(command, "ConfigureSlices")
        self.assertEqual(json.loads(argument), [{key: [float(value) for value in plane[key]] for key in plane}
                                                for plane in planes])
        for bad in (planes[:2], dict(planes=planes), [dict(plane, u=[0, 0, 0]) for plane in planes]):
            with self.subTest(bad=bad), self.assertRaises(urllib.error.HTTPError) as result:
                send(bad)
            self.assertEqual(result.exception.code, 400)
            result.exception.close()
        self.assertEqual(len(self.proxy.calls), 1)

    def test_a_pull_set_shares_settings_and_answers_for_the_device_that_took_a_scan(self):
        other = type(self.proxy)()
        self.proxy.state["active"] = dict(scan_id=11, revision=0)
        other.state["active"] = dict(scan_id=12, revision=0)
        control = ReconstructionControl(self.proxy, other)
        state = control.configure(dict(algorithm="fbp", filter="hann"))
        self.assertEqual(state["requested"]["options"]["filter"], "hann")
        self.assertEqual(state["active"]["scan_id"], 12)
        self.assertFalse(state["finished"])
        for proxy in (self.proxy, other):
            self.assertEqual(proxy.state["requested"]["revision"], 1)
        control.for_scan(12, 1)
        self.assertEqual(other.calls[-1], ("ReconstructionForScan", "12"))
        self.assertNotIn(("ReconstructionForScan", "12"), self.proxy.calls)

    def test_invalid_settings_do_not_reach_the_device_update(self):
        for options in ([], {"algorithm": "gridrec", "center": 65},
                        {"algorithm": "sirt", "min_constraint": 2, "max_constraint": 1},
                        {"algorithm": "fbp", "filter_cutoff": .5}):
            with self.subTest(options=options), self.assertRaises(urllib.error.HTTPError) as result:
                self.post(options)
            self.assertEqual(result.exception.code, 400)
            result.exception.close()
        self.assertTrue(all(command == "GetReconstruction" for command, _ in self.proxy.calls))
        self.assertEqual(self.proxy.state["requested"]["revision"], 0)

    def test_bad_json_and_cross_origin_posts_are_rejected(self):
        request = urllib.request.Request(self.url + "/api/reconstruction",
            data=b'{', headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as result:
            urllib.request.urlopen(request, timeout=3)
        self.assertEqual(result.exception.code, 400)
        result.exception.close()
        with self.assertRaises(urllib.error.HTTPError) as result:
            self.post(dict(algorithm="sirt"), Origin="http://elsewhere.invalid")
        self.assertEqual(result.exception.code, 403)
        result.exception.close()
        self.assertEqual(self.proxy.calls, [])

    def test_control_failure_keeps_history_available(self):
        self.history.append(np.zeros((1, 2, 3)), 40, 0, HEALTH)
        self.proxy.failure = "device unavailable"
        with self.get("/api/status") as response:
            state = json.load(response)
            self.assertEqual(state["viewer"]["received_volumes"], 1)
            self.assertEqual(state["reconstruction_control_error"], "device unavailable")
        with self.get("/api/volume") as response:
            self.assertEqual(response.status, 200)


class PipelineHTTPTests(unittest.TestCase):
    tearDown = HTTPTests.tearDown
    get = HTTPTests.get

    def setUp(self):
        HTTPTests.setUp(self)
        self.root = Path(self.directory.name)
        self.pipeline = PipelineControl(self.root)
        atomic_json(self.pipeline.state_path, running_state())
        self.server.RequestHandlerClass = make_handler(self.history, self.root,
                                                       pipeline_control=self.pipeline)

    def post(self, options, **headers):
        request = urllib.request.Request(self.url + "/api/pipeline", data=json.dumps(options).encode(),
            headers={"Content-Type": "application/json", **headers})
        return urllib.request.urlopen(request, timeout=3)

    def test_pending_restart_is_reported_without_changing_active_settings(self):
        with self.get("/api/pipeline") as response:
            self.assertEqual(json.load(response)["run_id"], 0)
        with self.post(dict(transport_batch=16, processing_mode="batched")) as response:
            state = json.load(response)
            self.assertEqual(state["requested"]["revision"], 1)
            self.assertEqual(state["active"]["revision"], 0)
        with self.get("/api/status") as response:
            self.assertEqual(json.load(response)["pipeline_control"], state)
        with self.assertRaises(urllib.error.HTTPError) as result:
            self.post(dict(network="tcp"))
        self.assertEqual(result.exception.code, 409)
        result.exception.close()

    def test_cross_origin_invalid_and_unavailable_controls_are_rejected(self):
        for options, headers, status in [({"gpu": True}, {}, 400),
                ({"transport_batch": MAX_BATCH + 1}, {}, 400), ({"unknown": 1}, {}, 400),
                ({"network": "tcp"}, {"Origin": "http://elsewhere.invalid"}, 403)]:
            with self.subTest(options=options), self.assertRaises(urllib.error.HTTPError) as result:
                self.post(options, **headers)
            self.assertEqual(result.exception.code, status)
            result.exception.close()
        self.assertFalse(self.pipeline.request_path.exists())
        self.server.RequestHandlerClass = make_handler(self.history, self.root)
        for method in (lambda: self.get("/api/pipeline"), lambda: self.post({"network": "tcp"})):
            with self.assertRaises(urllib.error.HTTPError) as result:
                method()
            self.assertEqual(result.exception.code, 503)
            result.exception.close()

    def test_stop_and_recommend_routes_use_separate_actions(self):
        calls = []
        def recommend(options):
            calls.append(("recommend", options))
            return dict(options=dict(running_state()["active"]["options"], receive_budget_mib=8),
                        information=dict(detector_frame_bytes=1024))
        def stop():
            calls.append(("stop",))
            return dict(running_state(), phase="stopping")
        self.pipeline.recommend = recommend
        self.pipeline.stop = stop
        def post_action(path, body, **headers):
            request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", **headers})
            return urllib.request.urlopen(request, timeout=3)
        with post_action("/api/pipeline/recommend", {"transport_batch": 16}) as response:
            self.assertEqual(json.load(response)["options"]["receive_budget_mib"], 8)
        self.assertFalse(self.pipeline.request_path.exists())
        for path in ("/api/pipeline/stop", "/api/pipeline/recommend"):
            with self.assertRaises(urllib.error.HTTPError) as result:
                post_action(path, {}, Origin="http://elsewhere.invalid")
            self.assertEqual(result.exception.code, 403)
            result.exception.close()
        with self.assertRaises(urllib.error.HTTPError) as result:
            post_action("/api/pipeline/stop", {"network": "tcp"})
        self.assertEqual(result.exception.code, 400)
        result.exception.close()
        with post_action("/api/pipeline/stop", {}) as response:
            self.assertEqual(json.load(response)["phase"], "stopping")
        self.assertEqual(calls, [("recommend", {"transport_batch": 16}), ("stop",)])

    def test_state_failure_is_reported_without_losing_the_volume(self):
        self.history.append(np.zeros((1, 2, 3)), 40, 0, HEALTH)
        self.pipeline.state_path.write_text("{")
        with self.get("/api/status") as response:
            state = json.load(response)
            self.assertEqual(state["viewer"]["received_volumes"], 1)
            self.assertIn("pipeline_control_error", state)
        with self.get("/api/volume") as response:
            self.assertEqual(response.status, 200)
        with self.assertRaises(urllib.error.HTTPError) as result:
            self.get("/api/pipeline")
        self.assertEqual(result.exception.code, 503)
        result.exception.close()


if __name__ == "__main__":
    unittest.main()
