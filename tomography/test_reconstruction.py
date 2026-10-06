"""Scientific orientation checks for analytic reconstruction, plus optional GPU checks."""
import os
import importlib.util
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from reconstruction import (SirtGPU, _gpu_arrays, _fbp_options, configuration, fbp_gpu,
                            gridrec, live_configuration, postprocess, reference_reconstruction, slice_blocks)


class LiveConfigurationTests(unittest.TestCase):
    def test_complete_settings_and_canonical_reports_round_trip(self):
        examples = [configuration("sirt", iterations=12, min_constraint=None, max_constraint=.03,
                                  slices_per_block=3),
                    configuration("fbp", "hann", filter_cutoff=.7, gaussian_fwhm=1, slices_per_block=2),
                    configuration("gridrec", "parzen", center=31.5, threads=2, scale_factor=2)]
        for options in examples:
            with self.subTest(options=options):
                self.assertEqual(live_configuration(options, 64), options)
        self.assertEqual(live_configuration({"algorithm": "sirt"}, 64), configuration("sirt"))

    def test_invalid_json_settings_and_inapplicable_knobs(self):
        for options in (None, [], {}, {"algorithm": "unknown"},
                        {"algorithm": "fbp", "unknown": 1},
                        {"algorithm": "fbp", "backend": "tomopy-cpu"},
                        {"algorithm": "gridrec", "center": 65},
                        {"algorithm": "gridrec", "center": True},
                        {"algorithm": "sirt", "iterations": 2.5},
                        {"algorithm": "sirt", "iterations": None},
                        {"algorithm": "sirt", "slices_per_block": -1},
                        {"algorithm": "fbp", "slices_per_block": 2.5},
                        {"algorithm": "gridrec", "slices_per_block": True},
                        {"algorithm": "gridrec", "slices_per_block": None},
                        {"algorithm": "sirt", "min_constraint": "none"},
                        {"algorithm": "sirt", "scale_factor": None},
                        {"algorithm": "fbp", "scale_factor": "2"},
                        {"algorithm": "fbp", "gaussian_fwhm": True},
                        {"algorithm": "fbp", "filter": "ram-lak", "filter_cutoff": .5},
                        {"algorithm": "gridrec", "relaxation": .5},
                        {"algorithm": "sirt", "min_constraint": 2, "max_constraint": 1},
                        {"algorithm": "fbp", "filter": ""},
                        {"algorithm": "fbp", "scale_factor": float("inf")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                live_configuration(options, 64)


class LiveProcessorTests(unittest.TestCase):
    def test_method_switching_and_knobs_reuse_sirt_scratch(self):
        class Array(np.ndarray):
            def get(self, stream=None):
                return np.asarray(self).copy()

            def set(self, value, stream=None):
                np.copyto(self, value)

        class Stream:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def synchronize(self): pass

        cupy = SimpleNamespace(dtype=np.dtype, float32=np.float32, copyto=np.copyto,
            empty=lambda shape, dtype: np.empty(shape, dtype).view(Array),
            cuda=SimpleNamespace(Device=lambda gpu: SimpleNamespace(use=lambda: None),
                                 ExternalStream=lambda pointer: Stream()))
        # Load the actual processor with CPU stand-ins only at GPU library boundaries.
        spec = importlib.util.spec_from_file_location("live_processor_test", Path(__file__).with_name("processors.py"))
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"cupy": cupy, "nvidia": SimpleNamespace(nvcomp=None)}):
            spec.loader.exec_module(module)
        source = np.ones((1, 2), np.float32).view(Array)
        output = np.empty((1, 2, 2), np.float32).view(Array)
        scan = dict(rows=1, columns=2, angles=2, theta=[0, 1], reconstruction=configuration("gridrec"))
        sirt = Mock()
        sirt.run.side_effect = lambda sino, out, iterations, **kw: out.fill(iterations)
        fbp = Mock(side_effect=lambda sino, out, theta, gpu, name, **kw: out.fill(3))
        grid = Mock(return_value=np.full(output.shape, 2, np.float32))
        with patch.object(module, "array_at", side_effect=lambda ptr, *args: source if ptr == 1 else output), \
             patch.object(module, "SirtGPU", return_value=sirt) as factory, \
             patch.object(module, "fbp_gpu", fbp), patch.object(module, "gridrec", grid):
            processor = module.Processor("reconstruct", scan, 0, 0, 40)
            self.assertFalse(hasattr(processor, "sirt"))
            examples = [configuration("gridrec", "parzen", center=1.5, threads=2),
                        configuration("sirt", iterations=12, relaxation=.5, min_constraint=None),
                        configuration("fbp", "hann", filter_cutoff=.7, scale_factor=2),
                        configuration("sirt", iterations=7, max_constraint=.02)]
            for options in examples:
                processor.begin_scan()
                processor.configure_reconstruction(options)
                processor.consume(1, source.nbytes, 0, 2, 0, 0)
                processor.consume(1, source.nbytes, 2, 2, 1, 1)
                processor.finish()
                expected = {"gridrec": 2, "sirt": options["iterations"], "fbp": 6}[options["algorithm"]]
                np.testing.assert_array_equal(output, expected)
            factory.assert_called_once()
            grid.assert_called_once_with(unittest.mock.ANY, processor.theta, "parzen", 2, center=1.5)
            self.assertEqual(fbp.call_args.kwargs["filter_cutoff"], .7)
            self.assertEqual(sirt.run.call_args_list[0].kwargs["relaxation"], .5)
            self.assertIsNone(sirt.run.call_args_list[0].kwargs["min_constraint"])
            self.assertEqual(sirt.run.call_args.kwargs["max_constraint"], .02)


class SliceBlockProcessorTests(unittest.TestCase):
    """Run the real one-worker assembly, replacing only GPU/library boundaries."""

    def load_processor(self):
        class Array(np.ndarray):
            def get(self, stream=None):
                return np.asarray(self).copy()

            def set(self, value, stream=None):
                np.copyto(self, value)

        class Stream:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def synchronize(self): pass

        cupy = SimpleNamespace(dtype=np.dtype, float32=np.float32, copyto=np.copyto,
            empty=lambda shape, dtype: np.empty(shape, dtype).view(Array),
            cuda=SimpleNamespace(Device=lambda gpu: SimpleNamespace(use=lambda: None),
                                 ExternalStream=lambda pointer: Stream()))
        spec = importlib.util.spec_from_file_location("block_processor_test", Path(__file__).with_name("processors.py"))
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"cupy": cupy, "nvidia": SimpleNamespace(nvcomp=None)}):
            spec.loader.exec_module(module)
        return module, Array

    def consume_scan(self, module, processor, sinogram, output, array_type):
        for projection, angle in enumerate(processor.theta):
            frame = np.ascontiguousarray(sinogram[:, projection]).view(array_type)
            with patch.object(module, "array_at", side_effect=lambda ptr, *args: frame if ptr == 1 else output):
                processor.consume(1, frame.nbytes, 2 if projection + 1 == len(processor.theta) else 0,
                                  2, projection, angle)
        processor.finish()

    def test_gridrec_blocks_match_whole_volume_including_gaussian_across_seams(self):
        module, array_type = self.load_processor()
        source, theta, _ = disk_scan(24)
        source = np.concatenate([source * np.float32(scale) for scale in (1, .2, .8, 0, .6)])
        scan = dict(rows=5, columns=24, angles=len(theta), theta=theta)
        options = configuration("gridrec", "hann", threads=2, gaussian_fwhm=2, scale_factor=1.3)
        expected = reference_reconstruction(source, theta, 0, options)
        sizes = []
        actual_gridrec = module.gridrec

        def reconstruct_block(sinogram, *args, **kwargs):
            sizes.append(sinogram.shape[0])
            return actual_gridrec(sinogram, *args, **kwargs)

        with patch.object(module, "gridrec", side_effect=reconstruct_block), \
             patch.object(module, "postprocess", side_effect=lambda volume, options, **kw: postprocess(volume, options)):
            for size in (0, 2, 9):
                options["slices_per_block"] = size
                processor = module.Processor("reconstruct", dict(scan, reconstruction=options), 0, 0, 40)
                output = np.full((5, 24, 24), np.nan, np.float32).view(array_type)
                self.consume_scan(module, processor, source, output, array_type)
                np.testing.assert_allclose(output, expected, rtol=3e-5, atol=2e-6)
                np.testing.assert_array_equal(processor.sinogram, source)
        self.assertEqual(sizes, [5, 2, 2, 1, 5])

    def test_sirt_reuses_regular_and_tail_workspaces_and_releases_old_layout(self):
        module, array_type = self.load_processor()
        source = np.arange(5 * 3 * 4, dtype=np.float32).reshape(5, 3, 4)
        options = configuration("sirt", slices_per_block=2)
        scan = dict(rows=5, columns=4, angles=3, theta=[0, 1, 2], reconstruction=options)
        runs = []

        class Runner:
            def __init__(self, shape, theta, gpu):
                self.shape = shape

            def run(self, sinogram, output, iterations, **kwargs):
                self_test.assertEqual(sinogram.shape, self.shape)
                self_test.assertTrue(sinogram.flags.c_contiguous and output.flags.c_contiguous)
                runs.append((id(self), sinogram.shape[0]))
                np.copyto(output, np.broadcast_to(sinogram.sum(axis=1)[:, None, :], output.shape))

        self_test = self
        with patch.object(module, "SirtGPU", side_effect=Runner) as factory:
            processor = module.Processor("reconstruct", scan, 0, 0, 40)
            for scale in (1, .5):
                processor.begin_scan()
                processor.configure_reconstruction(options)
                output = np.full((5, 4, 4), np.nan, np.float32).view(array_type)
                self.consume_scan(module, processor, source * scale, output, array_type)
                np.testing.assert_array_equal(output, np.broadcast_to((source * scale).sum(axis=1)[:, None, :], output.shape))
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(runs[:3], runs[3:])
            self.assertEqual(set(processor.sirt), {1, 2})
            processor.begin_scan()
            options = configuration("sirt", slices_per_block=3)
            processor.configure_reconstruction(options)
            self.assertFalse(hasattr(processor, "sirt"))
            self.consume_scan(module, processor, source, output, array_type)
            self.assertEqual(set(processor.sirt), {2, 3})
            self.assertEqual(factory.call_count, 4)

    def test_block_ranges_cover_every_slice_once(self):
        for rows, size, expected in ((5, 2, [(0, 2), (2, 4), (4, 5)]),
                                     (5, 0, [(0, 5)]), (5, 9, [(0, 5)])):
            self.assertEqual([(block.start, block.stop) for block in slice_blocks(rows, size)], expected)


def disk_scan(pixels=64, center_offset=0):
    """Exact parallel3d projections of asymmetric disks; no reconstruction code used."""
    theta = np.linspace(0, np.pi, 180, endpoint=False, dtype=np.float32)
    detector = np.arange(pixels, dtype=np.float32) + 0.5 - pixels / 2
    x, y = detector[None, :], detector[:, None]
    truth = np.zeros((pixels, pixels), dtype=np.float32)
    sinogram = np.zeros((1, len(theta), pixels), dtype=np.float32)
    # ASTRA parallel3d volume rows run in the positive y direction.
    for cx, cy, radius, intensity in ((-12, 10, 6, 1.0), (15, -13, 4, 0.6)):
        distance = detector[None, :] - center_offset - cx * np.cos(theta[:, None]) - cy * np.sin(theta[:, None])
        sinogram[0] += 2 * intensity * np.sqrt(np.maximum(radius**2 - distance**2, 0))
        truth += intensity * ((x - cx)**2 + (y - cy)**2 < radius**2)
    return sinogram, theta, truth


class GridRecTests(unittest.TestCase):
    def test_fractional_center_corrects_shift_without_mutating_input(self):
        for pixels in (64, 65):
            sinogram, theta, truth = disk_scan(pixels, center_offset=3.5)
            original = sinogram.copy()
            centered = gridrec(sinogram, theta, "ram-lak", 2, center=pixels / 2 + 3.5)
            uncorrected = gridrec(sinogram, theta, "ram-lak", 2)
            self.assertLess(np.linalg.norm(centered[0] - truth),
                            np.linalg.norm(uncorrected[0] - truth) * 0.5)
            np.testing.assert_array_equal(sinogram, original)

    def test_absolute_scale_orientation_and_power_of_two_padding(self):
        sinogram, theta, truth = disk_scan()
        original = sinogram.copy()
        volume = gridrec(sinogram, theta, "ram-lak", 2)
        self.assertEqual(volume.shape, (1, 64, 64))
        self.assertEqual(volume.dtype, np.float32)
        self.assertTrue(volume.flags.c_contiguous)
        self.assertLess(np.linalg.norm(volume[0] - truth) / np.linalg.norm(truth), 0.4)
        self.assertGreater(volume[0, 42, 20], 0.8)
        self.assertLess(abs(volume[0, 21, 20]), 0.1)
        np.testing.assert_array_equal(sinogram, original)

    def test_filters_and_odd_detector_width(self):
        sinogram, theta, truth = disk_scan(65)
        results = []
        for filter_name in ("ram-lak", "shepp-logan", "hann", "parzen"):
            volume = gridrec(sinogram, theta, filter_name, 2)
            self.assertEqual(volume.shape, (1, 65, 65))
            self.assertTrue(np.isfinite(volume).all())
            self.assertLess(np.linalg.norm(volume[0] - truth) / np.linalg.norm(truth), 0.5)
            results.append(volume)
        self.assertGreater(np.linalg.norm(results[0] - results[-1]), 0.1)


class TuningTests(unittest.TestCase):
    def test_defaults_preserve_existing_behavior(self):
        options = configuration("sirt")
        self.assertEqual((options["iterations"], options["relaxation"], options["min_constraint"]),
                         (40, 1, 0))
        self.assertIsNone(options["max_constraint"])
        volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        original = volume.copy()
        self.assertIs(postprocess(volume, options), volume)
        np.testing.assert_array_equal(volume, original)

    def test_invalid_and_inapplicable_settings_are_rejected(self):
        for options in ({"iterations": 2.5}, {"threads": 0}, {"relaxation": 0},
                        {"relaxation": 2}, {"relaxation": float("nan")},
                        {"min_constraint": 2, "max_constraint": 1},
                        {"max_constraint": float("inf")}, {"gaussian_fwhm": -1},
                        {"scale_factor": float("nan")}, {"scale_factor": 0},
                        {"filter_cutoff": 0.5}, {"center": 32}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                configuration("sirt", **options)
        for method, options in (("gridrec", {"filter_cutoff": 0.5}),
                                ("fbp", {"relaxation": 0.5}),
                                ("fbp", {"min_constraint": None}),
                                ("gridrec", {"max_constraint": 1}),
                                ("gridrec", {"center": float("nan")})):
            with self.subTest(method=method, options=options), self.assertRaises(ValueError):
                configuration(method, **options)
        for cutoff in (0, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                configuration("fbp", "hann", filter_cutoff=cutoff)
        self.assertEqual(_fbp_options(0, "hann", 0.6)["FilterD"], 0.6)
        self.assertNotIn("FilterD", _fbp_options(0, "ram-lak", None))

    def test_gaussian_fwhm_and_scaling(self):
        from scipy.ndimage import gaussian_filter

        volume = np.zeros((11, 11, 11), dtype=np.float32)
        volume[5, 5, 5] = 1
        expected = gaussian_filter(volume, sigma=1, mode="reflect") * 2
        options = configuration("fbp", gaussian_fwhm=np.sqrt(8 * np.log(2)), scale_factor=2)
        self.assertIs(postprocess(volume, options), volume)
        np.testing.assert_allclose(volume, expected, rtol=1e-6)
        self.assertAlmostEqual(float(volume.sum()), 2, places=6)
        self.assertGreater(volume[4, 5, 5], 0)

    def test_sirt_updates_match_matrix_equations(self):
        # Exercise the actual iteration loop with CPU stand-ins for CUDA projectors.
        matrix = np.array([[2, 1, 0, 0], [0, 2, 1, 0], [0, 0, 2, 1], [1, 0, 0, 2]], np.float32)
        source = np.array([1, -2, 3, 2], np.float32).reshape(1, 2, 2)
        projectors = SimpleNamespace(
            create=lambda config: 1, delete=lambda index: None,
            direct_FP=lambda index, value, out: np.copyto(out, (matrix @ value.ravel()).reshape(out.shape)),
            direct_BP=lambda index, value, out: np.copyto(out, (matrix.T @ value.ravel()).reshape(out.shape)))
        stream = SimpleNamespace(synchronize=lambda: None)
        cupy = SimpleNamespace(cuda=SimpleNamespace(
            Device=lambda gpu: nullcontext(), get_current_stream=lambda: stream,
            runtime=SimpleNamespace(deviceSynchronize=lambda: None)),
            subtract=np.subtract, maximum=np.maximum, minimum=np.minimum)
        for relaxation, minimum, maximum in ((0.5, None, None), (1.5, 0, 0.4), (1, -0.2, 0.8)):
            runner = object.__new__(SirtGPU)
            runner.shape, runner.theta, runner.gpu = source.shape, np.zeros(2), 0
            runner.projection_geometry = runner.volume_geometry = {}
            runner.ray_weights = (1 / matrix.sum(axis=1)).reshape(source.shape)
            runner.voxel_weights = (1 / matrix.sum(axis=0)).reshape(source.shape)
            runner.residual, runner.update = np.empty_like(source), np.empty_like(source)
            runner.weights_ready = True
            expected = np.zeros(4, np.float32)
            for _ in range(3):
                expected += relaxation * (matrix.T @ ((source.ravel() - matrix @ expected) /
                                                      matrix.sum(axis=1))) / matrix.sum(axis=0)
                if minimum is not None:
                    expected = np.maximum(expected, minimum)
                if maximum is not None:
                    expected = np.minimum(expected, maximum)
            output = np.full_like(source, 99)
            with patch.dict("sys.modules", cupy=cupy, astra=SimpleNamespace(projector3d=projectors)), \
                 patch("reconstruction._gpu_arrays"):
                runner.run(source, output, 3, relaxation=relaxation,
                           min_constraint=minimum, max_constraint=maximum)
            np.testing.assert_allclose(output.ravel(), expected, atol=1e-7)


class GPUArrayContractTests(unittest.TestCase):
    """Validate the residency contract using metadata only, without a CUDA device."""

    class GPUArray:
        def __init__(self, shape, dtype=np.float32, contiguous=True, gpu=0):
            self.shape, self.ndim, self.dtype = shape, len(shape), np.dtype(dtype)
            self.flags = SimpleNamespace(c_contiguous=contiguous)
            self.device = SimpleNamespace(id=gpu)

        def __array__(self, *args, **kwargs):
            raise AssertionError("GPU data must not be converted to NumPy")

    def setUp(self):
        cupy = SimpleNamespace(ndarray=self.GPUArray, float32=np.float32)
        self.cupy = patch.dict("sys.modules", cupy=cupy)
        self.cupy.start()
        self.addCleanup(self.cupy.stop)
        self.source = self.GPUArray((2, 3, 4))
        self.output = self.GPUArray((2, 4, 4))
        self.theta = np.zeros(3, dtype=np.float32)

    def test_host_input_or_output_cannot_silently_enable_staging(self):
        # An ASTRA stub with no methods ensures rejection happens before backend work.
        runner = object.__new__(SirtGPU)
        runner.theta, runner.gpu = self.theta, 0
        for source, output in ((np.zeros(self.source.shape), self.output),
                               (self.source, np.zeros(self.output.shape))):
            with self.subTest(source=type(source), output=type(output)):
                with patch.dict("sys.modules", astra=SimpleNamespace()):
                    with self.assertRaisesRegex(TypeError, "host staging is disabled"):
                        fbp_gpu(source, output, self.theta, 0, "ram-lak")
                    with self.assertRaisesRegex(TypeError, "host staging is disabled"):
                        runner.run(source, output, 4)

    def test_incompatible_dtype_layout_or_device_cannot_trigger_copies(self):
        for shape, extra in ((self.source.shape, {"dtype": np.float64}),
                             (self.source.shape, {"contiguous": False}),
                             (self.source.shape, {"gpu": 1})):
            bad_source = self.GPUArray(shape, **extra)
            bad_output = self.GPUArray(self.output.shape, **extra)
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    _gpu_arrays(bad_source, self.output, self.theta, 0)
                with self.assertRaises(ValueError):
                    _gpu_arrays(self.source, bad_output, self.theta, 0)

    def test_shapes_and_angles_must_match_without_converting_arrays(self):
        _gpu_arrays(self.source, self.output, self.theta, 0)
        for source, output, theta in (
            (self.GPUArray((3, 4)), self.output, self.theta),
            (self.GPUArray((2, 0, 4)), self.output, self.theta),
            (self.source, self.GPUArray((2, 4, 3)), self.theta),
            (self.source, self.output, self.theta[:-1]),
        ):
            with self.subTest(shape=source.shape, output=output.shape, angles=len(theta)):
                with self.assertRaises(ValueError):
                    _gpu_arrays(source, output, theta, 0)


@unittest.skipUnless(os.environ.get("TOMOGRAPHY_TEST_GPU") == "1", "set TOMOGRAPHY_TEST_GPU=1")
class FBPTests(unittest.TestCase):
    def test_cutoff_and_gpu_postprocessing_match_host_reference(self):
        import cupy as cp

        sinogram, theta, _ = disk_scan()
        source = cp.asarray(sinogram)
        output = cp.empty((1, 64, 64), dtype=cp.float32)
        options = configuration("fbp", "hann", filter_cutoff=0.6,
                                gaussian_fwhm=1.5, scale_factor=0.8)
        expected = reference_reconstruction(sinogram, theta, 0, options)
        with patch("cupy.asnumpy", side_effect=AssertionError("host conversion")):
            fbp_gpu(source, output, theta, 0, "hann", filter_cutoff=0.6)
            postprocess(output, options, gpu=True)
        np.testing.assert_allclose(output.get(), expected, rtol=3e-4, atol=2e-6)

    def test_gpu_links_agree_with_host_objects_and_preserve_orientation(self):
        import cupy as cp

        sinogram, theta, truth = disk_scan()
        source = cp.asarray(sinogram)
        output = cp.empty((1, 64, 64), dtype=cp.float32)
        for filter_name in ("ram-lak", "shepp-logan", "hann"):
            expected = reference_reconstruction(
                sinogram, theta, 0, configuration("fbp", filter_name))
            with patch("astra.data2d.create", side_effect=AssertionError("host data object")), \
                 patch("astra.data2d.get", side_effect=AssertionError("host download")), \
                 patch("cupy.asnumpy", side_effect=AssertionError("host conversion")):
                fbp_gpu(source, output, theta, 0, filter_name)
            result = output.get()
            np.testing.assert_allclose(result, expected, rtol=3e-4, atol=2e-6)
            self.assertLess(np.linalg.norm(result[0] - truth) / np.linalg.norm(truth), 0.3)
            np.testing.assert_array_equal(source.get(), sinogram)


@unittest.skipUnless(os.environ.get("TOMOGRAPHY_TEST_GPU") == "1", "set TOMOGRAPHY_TEST_GPU=1")
class SIRTTests(unittest.TestCase):
    def test_one_worker_block_assembly_matches_whole_geometry_and_reference(self):
        import cupy as cp
        from processors import Processor

        source, theta, _ = disk_scan(24)
        source = np.concatenate([source * np.float32(scale) for scale in (1, .2, .8, 0, .6)])
        stream = cp.cuda.Stream(non_blocking=True)
        for method in ("sirt", "fbp"):
            options = configuration(method, iterations=4, gaussian_fwhm=1.5, scale_factor=.8)
            expected = reference_reconstruction(source, theta, 0, options)
            results = []
            with stream:
                frames = [cp.asarray(np.ascontiguousarray(source[:, projection])) for projection in range(len(theta))]
                for size in (0, 2):
                    options = dict(options, slices_per_block=size)
                    scan = dict(rows=5, columns=24, angles=len(theta), theta=theta, reconstruction=options)
                    processor = Processor("reconstruct", scan, 0, stream.ptr, 4)
                    output = cp.empty((5, 24, 24), cp.float32)
                    for scan_number in range(2):
                        processor.begin_scan()
                        processor.configure_reconstruction(options)
                        scratch = {rows: tuple(array.data.ptr for array in (
                            runner.ray_weights, runner.voxel_weights, runner.residual, runner.update))
                            for rows, runner in getattr(processor, "sirt", {}).items()}
                        for projection, (frame, angle) in enumerate(zip(frames, theta)):
                            processor.consume(frame.data.ptr, frame.nbytes,
                                              output.data.ptr if projection + 1 == len(theta) else 0,
                                              2, projection, angle)
                        processor.finish()
                        stream.synchronize()
                        np.testing.assert_allclose(output.get(), expected, rtol=3e-4, atol=2e-6)
                        np.testing.assert_array_equal(processor.sinogram.get(), source)
                        if scan_number and method == "sirt":
                            self.assertEqual(scratch, {rows: tuple(array.data.ptr for array in (
                                runner.ray_weights, runner.voxel_weights, runner.residual, runner.update))
                                for rows, runner in processor.sirt.items()})
                    results.append(output.get())
            np.testing.assert_allclose(results[1], results[0], rtol=3e-4, atol=2e-6)

    def test_relaxation_and_bounds_match_independent_astra_steps(self):
        import cupy as cp

        sinogram, theta, _ = disk_scan()
        source = cp.asarray(sinogram)
        runner = SirtGPU(sinogram.shape, theta, 0)
        output = cp.empty((1, 64, 64), dtype=cp.float32)
        for relaxation, minimum, maximum in ((0.5, None, None), (1.5, 0, 0.25), (1, -0.1, 0.4)):
            options = configuration("sirt", iterations=4, relaxation=relaxation,
                                    min_constraint=minimum, max_constraint=maximum)
            expected = reference_reconstruction(sinogram, theta, 0, options)
            runner.run(source, output, 4, relaxation=relaxation,
                       min_constraint=minimum, max_constraint=maximum)
            np.testing.assert_allclose(output.get(), expected, rtol=3e-4, atol=2e-6)

    def test_repeated_scans_reuse_weights_and_scratch_with_distinct_output_slots(self):
        import astra
        import cupy as cp

        sinogram, theta, _ = disk_scan()
        sinogram = np.concatenate((sinogram, sinogram * np.float32(0.5)))
        iterations = 4
        runner = SirtGPU(sinogram.shape, theta, 0)
        scratch = (runner.ray_weights, runner.voxel_weights, runner.residual, runner.update)
        pointers = tuple(array.data.ptr for array in scratch)
        slots = [cp.empty((2, 64, 64), dtype=cp.float32) for _ in range(3)]
        for scan, (scale, output) in enumerate(zip((1, 0, 0.6), slots)):
            host = sinogram * np.float32(scale)
            expected = reference_reconstruction(host, theta, 0,
                                                 configuration("sirt", iterations=iterations))
            source = cp.asarray(host)
            with patch("astra.data3d.create", side_effect=AssertionError("host data object")), \
                 patch("astra.data3d.get", side_effect=AssertionError("host download")), \
                 patch("cupy.asnumpy", side_effect=AssertionError("host conversion")), \
                 patch("astra.projector3d.direct_FP", wraps=astra.projector3d.direct_FP) as fp, \
                 patch("astra.projector3d.direct_BP", wraps=astra.projector3d.direct_BP) as bp:
                runner.run(source, output, iterations)
                self.assertEqual(fp.call_count, iterations + (scan == 0))
                self.assertEqual(bp.call_count, iterations + (scan == 0))
            np.testing.assert_allclose(output.get(), expected, rtol=3e-4, atol=2e-6)
            np.testing.assert_array_equal(source.get(), host)
            self.assertEqual(tuple(array.data.ptr for array in (
                runner.ray_weights, runner.voxel_weights, runner.residual, runner.update)), pointers)
            self.assertTrue(runner.weights_ready)


if __name__ == "__main__":
    unittest.main()
