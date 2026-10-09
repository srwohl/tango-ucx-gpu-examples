"""Sliding-window updates and slice output: rules and geometry on the host, kernels on a GPU."""
import os
from types import SimpleNamespace
import unittest

import numpy as np

from host_buffering import memory_plan
from reconstruction import configuration, fbp_gpu
from streaming import (Updates, default_planes, plane_array, plane_list, publications, reference_slices,
                       slice_size, update_due, update_interval)
from test_reconstruction import disk_scan

OBLIQUE = [dict(origin=[1.7, 2.5, -3.0], u=[0.02, 0.9, 0.3], v=[0.01, -0.3, 0.9]),
           dict(origin=[-1.0, 12.2, 1.0], u=[0.2, 0.3, 0.1], v=[0.0, 0.2, 0.95]),
           dict(origin=[4, 23, 23], u=[-0.11, -0.5, 0], v=[0, -0.5, -0.9])]


def smooth_scan(rows, angles, columns, seed=0):
    """A sinogram without symmetry; smooth along the detector as real projections are."""
    sinogram = np.random.default_rng(seed).random((rows, angles, columns), dtype=np.float32)
    sinogram = sum(np.roll(sinogram, shift, -1) for shift in (-1, 0, 1, 2)) / 4
    return sinogram, np.linspace(0, np.pi, angles, endpoint=False, dtype=np.float32)


class UpdateRuleTests(unittest.TestCase):
    def test_the_first_rotation_publishes_once_and_later_scans_every_interval(self):
        angles = 12
        for interval, expected in ((4, [(0, 11), (1, 3), (1, 7), (1, 11), (2, 3), (2, 7), (2, 11)]),
                                   (angles, [(0, 11), (1, 11), (2, 11)])):
            received, due = 0, []
            for scan in range(3):
                for projection in range(angles):
                    received += 1
                    if update_due(received, projection, angles, interval):
                        due.append((scan, projection))
            self.assertEqual(due, expected)
            self.assertEqual(len(due), publications(3, angles, interval))
        self.assertEqual(publications(0, angles, 4), 0)
        self.assertEqual(update_interval({}, angles), angles)
        self.assertEqual(update_interval(dict(update_projections=4), angles), 4)

    def test_plan_accepts_only_layouts_the_ring_supports(self):
        scan = dict(rows=5, angles=12, columns=9)
        fbp = configuration("fbp")
        plan = memory_plan(dict(scan, buffering=dict(output_mode="slices", update_projections=4)), fbp)
        self.assertEqual((plan["slice_size"], plan["gpu_output_bytes"], plan["update_projections"]),
                         (9, 3 * 9 * 9 * 4, 4))
        self.assertEqual(memory_plan(dict(scan, buffering=dict(update_projections=6)),
                                     configuration("sirt"))["gpu_output_bytes"], 5 * 9 * 9 * 4)
        for buffering, options in ((dict(update_projections=5), fbp),
                                   (dict(update_projections=-4), fbp),
                                   (dict(update_projections=True), fbp),
                                   (dict(update_projections=4, sinogram_memory="host"), fbp),
                                   (dict(update_projections=4, output_mode="blocks", output_block_rows=5),
                                    dict(fbp, slices_per_block=5)),
                                   (dict(output_mode="slices", sinogram_memory="host"), fbp),
                                   (dict(output_mode="slices"), configuration("sirt")),
                                   (dict(output_mode="slices"), configuration("gridrec")),
                                   (dict(output_mode="slices"), configuration("fbp", gaussian_fwhm=1))):
            with self.subTest(buffering=buffering, algorithm=options["algorithm"]), self.assertRaises(ValueError):
                memory_plan(dict(scan, buffering=buffering), options)


class PlaneTests(unittest.TestCase):
    def test_planes_round_trip_and_reject_what_cannot_be_a_plane(self):
        planes = plane_array(OBLIQUE)
        self.assertEqual(planes.shape, (3, 3, 3))
        self.assertEqual(plane_list(planes), [{key: [float(value) for value in plane[key]]
                                               for key in ("origin", "u", "v")} for plane in OBLIQUE])
        np.testing.assert_array_equal(plane_array(planes), planes)
        good = OBLIQUE[0]
        for bad in (OBLIQUE[:2], OBLIQUE + OBLIQUE[:1], "planes", [good, good, dict(good, w=[0, 0, 1])],
                    [good, good, dict(origin=[0, 0, 0], u=[0, 1, 0])],
                    [good, good, dict(good, u=[0, 1])], [good, good, dict(good, u=[0, 1, "1"])],
                    [good, good, dict(good, u=[0, True, 0])], [good, good, dict(good, origin=[0, 0, float("nan")])],
                    [good, good, dict(good, origin=[0, 0, 1e7])], [good, good, dict(good, u=[0, 0, 0])],
                    [good, good, dict(origin=[0, 0, 0], u=[0, 1, 1], v=[0, 2, 2])], np.zeros((3, 3, 2))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plane_array(bad)

    def test_default_planes_are_the_central_slices_centred_in_their_images(self):
        for rows, columns in ((5, 12), (4, 11), (14, 6)):
            sinogram, theta = smooth_scan(rows, 20, columns)
            options, size = configuration("fbp", scale_factor=1.5), slice_size(rows, columns)
            # Every axial slice, placed at the image corner: the volume the planes cut through.
            volume = np.zeros((rows, columns, columns), np.float32)
            for z in range(rows):
                axial = dict(origin=[z, 0, 0], u=[0, 1, 0], v=[0, 0, 1])
                volume[z] = reference_slices(sinogram, theta, options, [axial] * 3)[0, :columns, :columns]
            self.assertGreater(np.abs(volume).min(), 0)
            z, c = (size - rows) // 2, (size - columns) // 2
            expected = np.zeros((3, size, size), np.float32)
            expected[0, c:c + columns, c:c + columns] = volume[rows // 2]
            expected[1, z:z + rows, c:c + columns] = volume[:, columns // 2, :]
            expected[2, z:z + rows, c:c + columns] = volume[:, :, columns // 2]
            np.testing.assert_allclose(reference_slices(sinogram, theta, options, default_planes(rows, columns)),
                                       expected, rtol=1e-6, atol=1e-7)

    def test_a_slice_recovers_the_density_of_exact_disk_projections(self):
        sinogram, theta, truth = disk_scan()
        image = reference_slices(sinogram, theta, configuration("fbp"), default_planes(1, 64))[0]
        self.assertLess(np.linalg.norm(image - truth) / np.linalg.norm(truth), 0.3)
        # Away from the edges, inside the unit disk and in the empty background.
        self.assertAlmostEqual(float(image[42 - 2:42 + 3, 20 - 2:20 + 3].mean()), 1.0, delta=0.03)
        self.assertAlmostEqual(float(image[2:8, 2:8].mean()), 0.0, delta=0.03)
        doubled = reference_slices(sinogram, theta, configuration("fbp", scale_factor=2), default_planes(1, 64))[0]
        np.testing.assert_allclose(doubled, 2 * image, rtol=1e-6)

    def test_a_tilted_plane_interpolates_between_detector_rows(self):
        sinogram, theta = smooth_scan(2, 20, 9)
        options = configuration("fbp")
        level = lambda z: dict(origin=[z, 0, 0], u=[0, 1, 0], v=[0, 0, 1])
        lower, upper, between = reference_slices(sinogram, theta, options, [level(0), level(1), level(.25)])
        np.testing.assert_allclose(between, .75 * lower + .25 * upper, rtol=1e-5, atol=1e-7)
        # Half a voxel beyond the last slice still belongs to it; further out is empty.
        edge, outside, _ = reference_slices(sinogram, theta, options, [level(1.5), level(1.51), level(0)])
        np.testing.assert_allclose(edge, upper, rtol=1e-6)
        self.assertFalse(outside.any())


class Subscription:
    """Batches of one publication, as a latest or every subscription returns them."""
    dtype = np.dtype([("index", "<u8"), ("scan_id", "<u8"), ("projection", "<u8")])

    def __init__(self, *identities):
        self.pending, self.outcome = list(identities), None

    def read(self, timeout=None):
        if not self.pending:
            self.outcome = "end"
            return None
        scan_id, projection = self.pending.pop(0)
        records = np.array([(0, scan_id, projection)], self.dtype)
        return SimpleNamespace(frames=1, records=records, array=np.full((1, 3, 2, 2), projection, np.float32))


class UpdatesTests(unittest.TestCase):
    def test_updates_share_a_scan_and_a_latest_reader_may_skip_some(self):
        updates = Updates(Subscription((7, 11), (8, 3), (8, 11), (10, 7)), 7, 12, 4)
        seen = []
        while (result := updates.read()) is not None:
            array, record = result
            self.assertEqual(array.shape, (3, 2, 2))
            seen.append((int(record["scan_id"]), int(record["projection"]), updates.complete(record)))
        self.assertEqual(seen, [(7, 11, True), (8, 3, False), (8, 11, True), (10, 7, False)])
        self.assertEqual(updates.outcome, "end")

    def test_repeated_earlier_or_misplaced_updates_are_refused(self):
        for identities in (((8, 3), (8, 3)), ((8, 7), (8, 3)), ((9, 3), (8, 11)), ((6, 11),),
                           ((7, 4),), ((7, 12),)):
            updates = Updates(Subscription(*identities), 7, 12, 4)
            with self.subTest(identities=identities), self.assertRaises(ValueError):
                while updates.read() is not None:
                    pass


@unittest.skipUnless(os.environ.get("TOMOGRAPHY_TEST_GPU") == "1", "set TOMOGRAPHY_TEST_GPU=1")
class GPUTests(unittest.TestCase):
    def test_kernel_matches_the_host_reference_and_astra_volume(self):
        import cupy as cp
        from streaming import SliceReconstructor

        for rows, angles, columns, name, cutoff in ((8, 96, 64, "ram-lak", None), (5, 45, 33, "hann", .6),
                                                    (40, 30, 24, "parzen", None)):
            sinogram, theta = smooth_scan(rows, angles, columns)
            options = configuration("fbp", name, filter_cutoff=cutoff, scale_factor=1.3)
            slices = SliceReconstructor(sinogram.shape, theta, cp)
            slices.configure(options)
            ring = cp.empty(sinogram.shape, cp.float32)
            for projection in range(angles):
                slices.store(ring, projection, cp.asarray(sinogram[:, projection]))
            output = cp.empty((3, slices.size, slices.size), cp.float32)
            for planes in (default_planes(rows, columns), OBLIQUE):
                slices.backproject(ring, planes, output)
                expected = reference_slices(sinogram, theta, options, planes)
                self.assertTrue(expected.any())
                np.testing.assert_allclose(output.get(), expected, rtol=0, atol=1e-5 * np.abs(expected).max())
            # ASTRA interpolates in its own way; the default planes are whole voxels of its volume.
            volume = cp.empty((rows, columns, columns), cp.float32)
            fbp_gpu(cp.asarray(sinogram), volume, theta, 0, name, filter_cutoff=cutoff)
            volume = volume.get() * np.float32(1.3)
            slices.backproject(ring, default_planes(rows, columns), output)
            z, c = (slices.size - rows) // 2, (slices.size - columns) // 2
            got, tolerance = output.get(), 3e-3 * np.abs(volume).max()
            np.testing.assert_allclose(got[0, c:c + columns, c:c + columns], volume[rows // 2], rtol=0, atol=tolerance)
            np.testing.assert_allclose(got[1, z:z + rows, c:c + columns], volume[:, columns // 2], rtol=0, atol=tolerance)
            np.testing.assert_allclose(got[2, z:z + rows, c:c + columns], volume[:, :, columns // 2], rtol=0, atol=tolerance)

    def test_worker_publishes_the_latest_rotation_as_slices_or_volumes(self):
        import cupy as cp
        from processors import Processor

        rows, angles, columns, interval = 5, 12, 16, 4
        theta = np.linspace(0, np.pi, angles, endpoint=False, dtype=np.float32)
        scans = [smooth_scan(rows, angles, columns, seed)[0] for seed in range(3)]
        stream = cp.cuda.Stream(non_blocking=True)
        size = slice_size(rows, columns)
        for mode in ("slices", "volume"):
            # The filter changes at the last scan: slices filter on arrival, volumes at each update.
            settings = [configuration("fbp"), configuration("fbp"), configuration("fbp", "hann", scale_factor=2)]
            scan = dict(rows=rows, columns=columns, angles=angles, theta=theta, reconstruction=settings[0],
                        buffering=dict(output_mode=mode, update_projections=interval))
            with stream:
                processor = Processor("reconstruct", scan, 0, stream.ptr, 4)
                output = cp.empty((3, size, size) if mode == "slices" else (rows, columns, columns), cp.float32)
                window, published = np.zeros_like(scans[0]), 0
                for number, (source, options) in enumerate(zip(scans, settings)):
                    processor.begin_scan()
                    processor.configure_reconstruction(options)
                    for projection in range(angles):
                        frame = cp.asarray(np.ascontiguousarray(source[:, projection]))
                        window[:, projection] = source[:, projection]
                        due = update_due(number * angles + projection + 1, projection, angles, interval)
                        output.fill(np.nan)
                        processor.consume(frame.data.ptr, frame.nbytes, output.data.ptr if due and mode == "volume" else 0,
                                          2, projection, theta[projection])
                        if mode == "slices" and not due:
                            with self.assertRaises(ValueError):
                                processor.reconstruct_slices(output.data.ptr, plane_array(OBLIQUE).ravel().tolist())
                        if not due:
                            stream.synchronize()
                            self.assertTrue(np.isnan(output.get()).all())
                            continue
                        published += 1
                        if mode == "slices":
                            processor.reconstruct_slices(output.data.ptr, plane_array(OBLIQUE).ravel().tolist())
                            mixed = number == 2 and projection + 1 < angles
                            expected = None if mixed else reference_slices(window, theta, options, OBLIQUE)
                        else:
                            expected = cp.empty_like(output)
                            fbp_gpu(cp.asarray(window), expected, theta, 0, options["filter"])
                            expected = expected.get() * np.float32(options["scale_factor"])
                        stream.synchronize()
                        self.assertTrue(np.isfinite(output.get()).all())
                        if expected is not None:
                            np.testing.assert_allclose(output.get(), expected, rtol=0,
                                                       atol=1e-5 * np.abs(expected).max())
                    processor.finish()
                self.assertEqual(published, publications(3, angles, interval))
                self.assertGreater(processor.reconstruct_ns, 0)


if __name__ == "__main__":
    unittest.main()
