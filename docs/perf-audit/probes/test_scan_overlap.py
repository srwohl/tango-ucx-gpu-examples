"""GPU correctness checks for the experiment's grouped, independently owned scans."""
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tomography"))

import cupy as cp
import numpy as np

from reconstruction import configuration, reference_reconstruction
from scan_overlap import ScanAssembly


class ScanAssemblyTests(unittest.TestCase):
    def setUp(self):
        cp.cuda.Device(0).use()
        self.rows, self.angles, self.columns = 4, 31, 65
        self.theta = np.linspace(0, np.pi, self.angles, endpoint=False, dtype=np.float32)
        self.options = configuration("fbp", "hann")
        self.scan = dict(rows=self.rows, angles=self.angles, columns=self.columns,
                         theta=self.theta.tolist(), reconstruction=self.options)
        self.assembly = cp.cuda.Stream(non_blocking=True)
        self.reconstruction = cp.cuda.Stream(non_blocking=True)
        self.worker = ScanAssembly(self.scan, 0, self.assembly.ptr, self.reconstruction.ptr, 1, 2)
        self.host = np.random.default_rng(42).normal(size=(self.angles, self.rows, self.columns)).astype(np.float32)
        self.source = cp.asarray(self.host)
        cp.cuda.get_current_stream().synchronize()
        self.frame_bytes = self.rows * self.columns * 4

    def tearDown(self):
        self.worker.drain()

    def metadata(self, source, start, stop):
        return [(source.data.ptr + index * self.frame_bytes, self.frame_bytes, 2,
                 index, float(self.theta[index])) for index in range(start, stop)]

    def assemble(self, buffer, scan_id, source, options):
        self.worker.begin_scan(buffer, scan_id, options)
        for start in range(0, self.angles, 7):
            self.worker.consume_group(buffer, self.metadata(source, start, min(start + 7, self.angles)))

    def test_invalid_groups_do_not_mutate_scan(self):
        self.worker.begin_scan(0, 10, self.options)
        with self.assembly:
            self.worker.processors[0].sinogram.fill(cp.nan)
        self.assembly.synchronize()
        before = self.worker.processors[0].sinogram.get()
        originals = self.metadata(self.source, 0, 2)
        mutations = ((0, 4, float(self.theta[0]) + 1), (1, 3, 3),
                     (0, 1, self.frame_bytes - 4), (0, 2, 1),
                     (1, 0, originals[1][0] + 4))
        for frame, field, value in mutations:
            with self.subTest(field=field):
                entries = [list(entry) for entry in originals]
                entries[frame][field] = value
                with self.assertRaises(ValueError):
                    self.worker.consume_group(0, entries)
                self.assertEqual(self.worker.processors[0].projections, 0)
                np.testing.assert_array_equal(self.worker.processors[0].sinogram.get(), before)
        np.testing.assert_array_equal(self.source.get(), self.host)

    def test_incomplete_and_oversized_scans_are_rejected(self):
        self.worker.begin_scan(0, 10, self.options)
        self.worker.consume_group(0, self.metadata(self.source, 0, 2))
        output = cp.empty((self.rows, self.columns, self.columns), dtype=cp.float32)
        with self.assertRaises(ValueError):
            self.worker.reconstruct(0, output.data.ptr)
        oversized = self.metadata(self.source, 2, self.angles)
        oversized.append(oversized[-1])
        with self.assertRaises(ValueError):
            self.worker.consume_group(0, oversized)
        self.assertEqual(self.worker.processors[0].projections, 2)

    def test_different_scans_settings_and_buffer_reuse_match_independent_reference(self):
        second_host = self.host * np.float32(-0.3) + np.float32(0.2)
        second_options = configuration("fbp", "parzen", gaussian_fwhm=0.8, scale_factor=1.2)
        first_expected = reference_reconstruction(self.host.transpose(1, 0, 2), self.theta, 0, self.options)
        second_expected = reference_reconstruction(second_host.transpose(1, 0, 2), self.theta, 0, second_options)
        second_source = cp.asarray(second_host)
        outputs = [cp.empty((self.rows, self.columns, self.columns), dtype=cp.float32) for _ in range(2)]
        cp.cuda.get_current_stream().synchronize()
        self.assemble(0, 10, self.source, self.options)
        ready = cp.cuda.Event()
        ready.record(self.assembly)
        self.reconstruction.wait_event(ready)
        failures = []

        def reconstruct_first():
            try:
                self.worker.reconstruct(0, outputs[0].data.ptr)
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=reconstruct_first)
        thread.start()
        try:
            self.assemble(1, 11, second_source, second_options)
        finally:
            thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        ready.record(self.assembly)
        self.reconstruction.wait_event(ready)
        self.worker.reconstruct(1, outputs[1].data.ptr)
        self.worker.drain()
        np.testing.assert_allclose(outputs[0].get(), first_expected, rtol=3e-4, atol=2e-6)
        np.testing.assert_allclose(outputs[1].get(), second_expected, rtol=3e-4, atol=2e-6)
        self.assemble(0, 12, second_source, second_options)
        ready.record(self.assembly)
        self.reconstruction.wait_event(ready)
        self.worker.reconstruct(0, outputs[0].data.ptr)
        self.worker.drain()
        np.testing.assert_allclose(outputs[0].get(), second_expected, rtol=3e-4, atol=2e-6)
        np.testing.assert_array_equal(self.source.get(), self.host)
        np.testing.assert_array_equal(second_source.get(), second_host)


if __name__ == "__main__":
    unittest.main()
