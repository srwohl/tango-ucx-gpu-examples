"""TomocuPy settings on the host, plus optional GPU checks against ASTRA FBP."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import reconstruction
from host_buffering import memory_plan
from reconstruction import (TOMOCUPY_ALGORITHMS, TOMOCUPY_FILTERS, available_algorithms, configuration,
                            live_configuration, tomocupy_geometry)
from tomocupy_backend import chunk_rows, padded_size, scratch_bytes

BUILT = set(TOMOCUPY_ALGORITHMS) <= set(available_algorithms())


class Built:
    """Settings checks must not depend on whether this checkout has built the modules."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        (Path(directory.name) / "tomocupy").mkdir()
        (Path(directory.name) / "tomocupy" / "_cfunc_lprec.so").touch()
        environment = patch.dict(os.environ, TOMOCUPY_BUILD=directory.name)
        environment.start()
        self.addCleanup(environment.stop)


class SettingsTests(Built, unittest.TestCase):
    def test_defaults_knobs_and_canonical_round_trip(self):
        for method in TOMOCUPY_ALGORITHMS:
            options = configuration(method)
            self.assertEqual((options["backend"], options["filter"], options["dtype"], options["center"]),
                             ("tomocupy-cuda", "ram-lak", "float32", None))
            self.assertEqual(live_configuration({"algorithm": method}, 64), options)
            tuned = configuration(method, "cosine2", center=31.25, dtype="float16", gaussian_fwhm=1,
                                  scale_factor=2, slices_per_block=4)
            self.assertEqual(live_configuration(tuned, 64), tuned)
        self.assertEqual(set(TOMOCUPY_FILTERS) - set(reconstruction.FILTERS), {"hamming", "cosine", "cosine2"})
        # Other methods report no precision and keep their canonical settings.
        self.assertIsNone(configuration("fbp")["dtype"])
        self.assertEqual(live_configuration(configuration("gridrec"), 64), configuration("gridrec"))

    def test_inapplicable_knobs_are_rejected(self):
        for arguments, keywords in ((("fbp", "hamming"), {}), (("gridrec", "cosine"), {}),
                                    (("fbp",), dict(dtype="float32")), (("sirt",), dict(dtype="float16")),
                                    (("lprec",), dict(dtype="float64")), (("lprec", "none"), {}),
                                    (("linerec", "hann"), dict(filter_cutoff=.5)),
                                    (("fourierrec",), dict(relaxation=.5)), (("fbp",), dict(center=3))):
            with self.subTest(arguments=arguments, keywords=keywords), self.assertRaises(ValueError):
                configuration(*arguments, **keywords)
        for options in ({"algorithm": "lprec", "dtype": 16}, {"algorithm": "lprec", "center": 65},
                        {"algorithm": "fourierrec", "filter": "ramp"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                live_configuration(options, 64)

    def test_geometry_each_method_cannot_reconstruct(self):
        theta = np.linspace(0, np.pi, 90, endpoint=False)
        for method in TOMOCUPY_ALGORITHMS:
            tomocupy_geometry(configuration(method), 64, theta)
            with self.assertRaisesRegex(ValueError, "32 detector columns"):
                tomocupy_geometry(configuration(method), 31)
        tomocupy_geometry(configuration("fbp"), 5, theta[::-1])
        with self.assertRaisesRegex(ValueError, "even detector width"):
            live_configuration({"algorithm": "fourierrec"}, 65)
        tomocupy_geometry(configuration("lprec"), 65, theta)
        tomocupy_geometry(configuration("linerec", dtype="float16"), 100, theta)
        for method in ("fourierrec", "lprec"):
            with self.assertRaisesRegex(ValueError, "power-of-two"):
                live_configuration({"algorithm": method, "dtype": "float16"}, 100)
        irregular = theta.copy()
        irregular[7] += .01
        for angles in (irregular, np.linspace(0, 2 * np.pi, 90, endpoint=False), theta + .1, theta[:1]):
            with self.assertRaisesRegex(ValueError, "equally spaced"):
                tomocupy_geometry(configuration("lprec"), 64, angles)
            tomocupy_geometry(configuration("linerec"), 64, angles)

    def test_memory_plan_reports_scratch_and_checks_the_scan(self):
        scan = dict(rows=8, angles=96, columns=64, theta=np.linspace(0, np.pi, 96, endpoint=False).tolist())
        self.assertEqual(memory_plan(scan, configuration("fbp"))["gpu_backend_scratch_bytes"], 0)
        plan = memory_plan(scan, configuration("lprec"))
        self.assertEqual(plan["gpu_backend_scratch_bytes"], scratch_bytes("lprec", 8, 96, 64))
        self.assertGreater(plan["gpu_backend_scratch_bytes"], 2 * 8 * 96 * 256 * 4)
        half = memory_plan(scan, configuration("lprec", dtype="float16", slices_per_block=4))
        self.assertLess(half["gpu_backend_scratch_bytes"], plan["gpu_backend_scratch_bytes"])
        with self.assertRaisesRegex(ValueError, "equally spaced"):
            memory_plan(dict(scan, theta=scan["theta"][::-1]), configuration("lprec"))
        with self.assertRaisesRegex(ValueError, "even detector width"):
            memory_plan(dict(scan, columns=65), configuration("fourierrec"))
        with self.assertRaisesRegex(ValueError, "slice output uses FBP"):
            memory_plan(dict(scan, buffering=dict(output_mode="slices")), configuration("lprec"))

    def test_chunks_are_bounded_and_fit_each_method(self):
        self.assertEqual((padded_size(100), padded_size(100, "float16"), padded_size(256, "float16")),
                         (400, 512, 1024))
        with patch("tomocupy_backend.FBP_FILTER_SAMPLES", 3 * 90 * 256):
            self.assertEqual([chunk_rows(method, 8, 90, 64) for method in TOMOCUPY_ALGORITHMS], [4, 3, 3])
            self.assertEqual([chunk_rows(method, 1, 90, 64) for method in TOMOCUPY_ALGORITHMS], [2, 1, 2])
        with patch("tomocupy_backend.FBP_FILTER_SAMPLES", 1):
            self.assertEqual([chunk_rows(method, 8, 90, 64) for method in TOMOCUPY_ALGORITHMS], [2, 1, 2])


class AvailabilityTests(unittest.TestCase):
    def test_unbuilt_methods_are_not_offered_or_accepted(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, TOMOCUPY_BUILD=directory):
            self.assertEqual(available_algorithms(), ("sirt", "fbp", "gridrec"))
            with self.assertRaisesRegex(ValueError, "build-tomocupy"):
                live_configuration({"algorithm": "lprec"}, 64)
            self.assertEqual(live_configuration({"algorithm": "fbp"}, 64), configuration("fbp"))


def discs(rows, angles, columns, theta):
    """An analytic sinogram of off-centre discs that grow along z."""
    detector = (np.arange(columns, dtype=np.float32) - (columns - 1) / 2)[None, :]
    cosine, sine = np.cos(theta)[:, None], np.sin(theta)[:, None]
    sinogram = np.zeros((rows, angles, columns), dtype=np.float32)
    for row in range(rows):
        for cx, cy, radius, density in ((0, 0, .38, 1), (.17, .09, .11, .8), (-.21, .12, .07, -.5),
                                        (.05, -.24, .05, 1.5)):
            radius *= columns * (1 + .1 * row / rows)
            offset = detector - columns * (cx * cosine + cy * sine)
            sinogram[row] += .02 * density * np.sqrt(np.maximum(radius**2 - offset**2, 0))
    return sinogram


@unittest.skipUnless(os.environ.get("TOMOGRAPHY_TEST_GPU") == "1" and BUILT,
                     "set TOMOGRAPHY_TEST_GPU=1 and run `pixi run build-tomocupy`")
class GPUTests(unittest.TestCase):
    def compare(self, method, rows, angles, columns, limit, *, roll=0, **keywords):
        """Relative L2 difference from ASTRA FBP inside the reconstruction circle."""
        import cupy as cp
        from tomocupy_backend import TomocupyGPU

        theta = np.linspace(0, np.pi, angles, endpoint=False, dtype=np.float32)
        host = discs(rows, angles, columns, theta)
        expected = cp.empty((rows, columns, columns), dtype=cp.float32)
        reconstruction.fbp_gpu(cp.asarray(host), expected, theta, 0, "ram-lak")
        shifted = np.roll(host, roll, axis=2)
        source = cp.asarray(shifted)
        # Borrowed publisher memory holds whatever the last volume left there.
        slot = cp.full((rows + 2, columns, columns), np.nan, dtype=cp.float32)
        engine = TomocupyGPU(method, theta, columns, rows, 0, center=columns / 2 + roll, **keywords)
        for _ in range(2):
            engine.run(source, slot[1:-1])
        actual, expected = slot[1:-1].get(), expected.get()
        self.assertTrue(np.isnan(slot[[0, -1]].get()).all())
        np.testing.assert_array_equal(source.get(), shifted)
        grid = np.arange(columns) - (columns - 1) / 2
        inside = grid[:, None]**2 + grid[None, :]**2 < (.45 * columns)**2
        error = np.linalg.norm((actual - expected)[:, inside]) / np.linalg.norm(expected[:, inside])
        self.assertLess(error, limit, (method, rows, angles, columns, keywords))
        return engine, actual

    def test_orientation_scale_and_grid_match_astra(self):
        # LineRec with an odd width needs no resampling, so it isolates the convention.
        self.compare("linerec", 4, 181, 129, 1e-3)
        self.compare("linerec", 4, 181, 129, 1e-3, roll=-5)
        for method, limit in (("fourierrec", .04), ("lprec", .035), ("linerec", .025)):
            with self.subTest(method=method):
                self.compare(method, 4, 360, 256, limit)
                self.compare(method, 4, 360, 256, limit, roll=7)
        self.compare("lprec", 4, 181, 129, .04, roll=3)

    def test_short_last_block_single_slice_and_half_precision(self):
        for method in TOMOCUPY_ALGORITHMS:
            with self.subTest(method=method):
                with patch("tomocupy_backend.FBP_FILTER_SAMPLES", 4 * 180 * 512):
                    engine, _ = self.compare(method, 7, 180, 128, .05)
                    self.assertEqual(engine.rows, 4)
                self.compare(method, 1, 180, 128, .05)
                self.compare(method, 4, 360, 256, .05, dtype="float16")
        self.compare("linerec", 4, 180, 200, .05, dtype="float16")

    def test_filters_change_the_volume_and_the_reference_repeats_it(self):
        from tomocupy_backend import reference

        theta = np.linspace(0, np.pi, 180, endpoint=False, dtype=np.float32)
        host = discs(6, 180, 128, theta)
        for method in TOMOCUPY_ALGORITHMS:
            with self.subTest(method=method):
                volumes = {}
                for name in TOMOCUPY_FILTERS:
                    options = configuration(method, name, center=63.5)
                    volumes[name] = reference(host, theta, 0, options)
                    self.assertTrue(np.isfinite(volumes[name]).all())
                    np.testing.assert_allclose(reference(host, theta, 0, options), volumes[name],
                                               rtol=3e-4, atol=2e-6)
                for name in set(TOMOCUPY_FILTERS) - {"ram-lak"}:
                    self.assertGreater(np.abs(volumes[name] - volumes["ram-lak"]).max(), 1e-4)
                # FourierRec transforms slices in pairs and each takes a little of its partner,
                # so its volume follows the block layout. The other methods' slices are independent.
                whole = reference(host, theta, 0, configuration(method))
                blocks = reference(host, theta, 0, configuration(method, slices_per_block=4))
                difference = np.abs(blocks - whole).max() / np.abs(whole).max()
                if method == "fourierrec":
                    self.assertTrue(1e-4 < difference < .03, difference)
                else:
                    self.assertLess(difference, 1e-5)

    def test_processor_switches_methods_and_releases_the_engine(self):
        import cupy as cp
        from processors import Processor

        theta = np.linspace(0, np.pi, 96, endpoint=False, dtype=np.float32)
        host = discs(8, 96, 64, theta)
        scan = dict(rows=8, columns=64, angles=96, theta=theta.tolist(),
                    reconstruction=configuration("lprec"))
        stream = cp.cuda.Stream(non_blocking=True)
        processor = Processor("reconstruct", scan, 0, stream.ptr, 40)
        processor.sinogram[...] = cp.asarray(host)
        output = cp.empty((8, 64, 64), dtype=cp.float32)
        with processor.stream:
            processor.reconstruct(output)
            engine = processor.tomocupy
            processor.reconstruct(output)
            self.assertIs(processor.tomocupy, engine)
            np.testing.assert_allclose(output.get(), reconstruction.reference_reconstruction(
                host, theta, 0, scan["reconstruction"]), rtol=3e-4, atol=2e-6)
            for options in (configuration("lprec", "hamming"), configuration("linerec", "hamming"),
                            configuration("linerec", "hamming", slices_per_block=3),
                            configuration("fourierrec", center=31, dtype="float16", scale_factor=2)):
                processor.configure_reconstruction(options)
                self.assertFalse(hasattr(processor, "tomocupy"))
                processor.reconstruct(output)
                expected = reconstruction.reference_reconstruction(host, theta, 0, options)
                # Half precision FourierRec sums in no fixed order.
                atol = 5e-3 * np.abs(expected).max() if options["dtype"] == "float16" else 2e-6
                np.testing.assert_allclose(output.get(), expected, rtol=3e-4, atol=atol)
            processor.configure_reconstruction(configuration("fbp"))
            self.assertFalse(hasattr(processor, "tomocupy"))


if __name__ == "__main__":
    unittest.main()
