"""HDF5 import and compressed-frame checks using small files; no GPU required."""
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import lz4.block
import numpy as np

from demo import parse_args, reconstruction_options, streaming_rates, workload_description
from scan import from_hdf5, hdf5_dimensions, selection


class HDF5ScanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.file = self.root / "input.h5"
        self.raw = (np.arange(5 * 4 * 6).reshape(5, 4, 6) + 100).astype(np.uint16)
        with h5py.File(self.file, "w") as file:
            file["exchange/data"] = self.raw
            file["exchange/data_dark"] = np.stack([np.full((4, 6), 10), np.full((4, 6), 14)])
            file["exchange/data_white"] = np.stack([np.full((4, 6), 400), np.full((4, 6), 402)])
            file["exchange/theta"] = [0, 30, 60, 90, 120]

    def prepare(self, **options):
        return from_hdf5(self.root / "scan", self.file, **options)

    def test_selected_counts_angles_and_reference_survive_compression(self):
        meta = self.prepare(sino=selection("1:4:2"), proj=selection("1:5:2"))
        self.assertEqual((meta["rows"], meta["columns"], meta["angles"]), (2, 6, 2))
        self.assertEqual(meta["element"], "u16")
        self.assertEqual(meta, json.loads((self.root / "scan/scan.json").read_text()))
        np.testing.assert_allclose(meta["theta"], np.deg2rad([30, 90]), rtol=1e-7)
        raw = self.raw[1:5:2, 1:4:2, :]
        dark, flat = np.full((2, 6), 12, np.uint16), np.full((2, 6), 401, np.uint16)
        expected_frames = [dark, flat, *raw]
        payloads = (self.root / "scan/compressed.bin").read_bytes()
        for index, (record, expected) in enumerate(zip(meta["frames"], expected_frames)):
            compressed = payloads[record["offset"]:record["offset"] + record["bytes"]]
            frame = np.frombuffer(lz4.block.decompress(compressed, uncompressed_size=24),
                                  np.uint16).reshape(2, 6)
            np.testing.assert_array_equal(frame, expected)
            self.assertEqual(record["kind"], min(index, 2))
            self.assertEqual(record["projection"], max(0, index - 2))
        with np.load(self.root / "scan/reference.npz") as reference:
            self.assertNotIn("phantom", reference)
            corrected = -np.log(np.clip((raw.astype(np.float64) - 12) / 389, 1e-6, 1))
            np.testing.assert_array_equal(reference["sinogram"], corrected.transpose(1, 0, 2).astype(np.float32))
            np.testing.assert_array_equal(reference["dark"], dark)
            np.testing.assert_array_equal(reference["flat"], flat)

    def test_radian_metadata_and_custom_paths(self):
        with h5py.File(self.file, "a") as file:
            file.move("exchange", "custom")
            del file["custom/theta"]
            file["custom/theta"] = np.linspace(0, 2, 5)
            file["custom/theta"].attrs["units"] = "radians"
        meta = self.prepare(data_path="custom/data", flat_path="custom/data_white",
                            dark_path="custom/data_dark", theta_path="custom/theta")
        np.testing.assert_array_equal(meta["theta"], np.linspace(0, 2, 5))
        self.assertEqual(meta["source"]["theta_units"], "radians")

    def test_explicit_units_override_metadata(self):
        with h5py.File(self.file, "a") as file:
            file["exchange/theta"].attrs["units"] = "unsupported"
        meta = self.prepare(theta_units="radians")
        np.testing.assert_array_equal(meta["theta"], [0, 30, 60, 90, 120])

    def test_missing_theta_generates_full_scan_angles_before_selection(self):
        with h5py.File(self.file, "a") as file:
            del file["exchange/theta"]
        meta = self.prepare(proj=selection("1:5:2"))
        np.testing.assert_allclose(meta["theta"], np.linspace(0, np.pi, 5)[1:5:2], rtol=1e-7)
        self.assertTrue(meta["source"]["generated_theta"])

    def test_single_calibration_images_and_clipping(self):
        with h5py.File(self.file, "a") as file:
            for path, value in [("data_dark", 10), ("data_white", 400)]:
                del file[f"exchange/{path}"]
                file[f"exchange/{path}"] = np.full((4, 6), value, np.uint16)
            file["exchange/data"][0] = 0
            file["exchange/data"][1] = 500
        self.prepare()
        with np.load(self.root / "scan/reference.npz") as reference:
            np.testing.assert_allclose(reference["sinogram"][:, 0, :], -np.log(1e-6), rtol=1e-7)
            np.testing.assert_array_equal(reference["sinogram"][:, 1, :], 0)

    def test_fractional_calibration_stack_preserves_mean_in_float_frames(self):
        with h5py.File(self.file, "a") as file:
            del file["exchange/data_dark"]
            del file["exchange/data_white"]
            # Neither the individual values nor the calibration mean is rounded
            # to integer counts when the detector stream uses float32.
            file["exchange/data_dark"] = np.stack([
                np.full((4, 6), 10.4, np.float32), np.full((4, 6), 10.8, np.float32)])
            file["exchange/data_white"] = np.stack([
                np.full((4, 6), 400.2, np.float64), np.full((4, 6), 400.4, np.float64)])
        meta = self.prepare(sino=selection("1:4:2"))
        self.assertEqual(meta["rows"], 2)
        self.assertEqual(meta["element"], "f32")
        dark = np.float32(np.mean(np.array([10.4, 10.8], np.float32), dtype=np.float64))
        flat = np.float32(400.3)
        with np.load(self.root / "scan/reference.npz") as reference:
            np.testing.assert_array_equal(reference["dark"], np.full((2, 6), dark, np.float32))
            np.testing.assert_array_equal(reference["flat"], np.full((2, 6), flat, np.float32))
            corrected = -np.log(np.clip((self.raw[:, 1:4:2, :].astype(np.float64) - float(dark)) /
                                        (float(flat) - float(dark)), 1e-6, 1))
            np.testing.assert_array_equal(reference["sinogram"], corrected.transpose(1, 0, 2).astype(np.float32))
        payloads = (self.root / "scan/compressed.bin").read_bytes()
        for record, value in zip(meta["frames"][:2], (dark, flat)):
            compressed = payloads[record["offset"]:record["offset"] + record["bytes"]]
            decoded = np.frombuffer(lz4.block.decompress(compressed, uncompressed_size=48), np.float32)
            np.testing.assert_array_equal(decoded, np.full(12, value, np.float32))

    def test_fractional_single_calibration_images_are_preserved(self):
        with h5py.File(self.file, "a") as file:
            for path, value in [("data_dark", 10.6), ("data_white", 400.2)]:
                del file[f"exchange/{path}"]
                file[f"exchange/{path}"] = np.full((4, 6), value, np.float32)
        self.prepare()
        with np.load(self.root / "scan/reference.npz") as reference:
            np.testing.assert_array_equal(reference["dark"], np.full((4, 6), 10.6, np.float32))
            np.testing.assert_array_equal(reference["flat"], np.full((4, 6), 400.2, np.float32))

    def test_fractional_projections_survive_compression_and_reference_correction(self):
        raw = self.raw.astype(np.float32) + np.float32(.25)
        with h5py.File(self.file, "a") as file:
            del file["exchange/data"]
            file["exchange/data"] = raw
        meta = self.prepare(sino=selection("1:4:2"), proj=selection("1:5:2"))
        self.assertEqual(meta["element"], "f32")
        expected_frames = [np.full((2, 6), 12, np.float32), np.full((2, 6), 401, np.float32),
                           *raw[1:5:2, 1:4:2, :]]
        payloads = (self.root / "scan/compressed.bin").read_bytes()
        for record, expected in zip(meta["frames"], expected_frames):
            compressed = payloads[record["offset"]:record["offset"] + record["bytes"]]
            frame = np.frombuffer(lz4.block.decompress(compressed, uncompressed_size=48),
                                  np.float32).reshape(2, 6)
            np.testing.assert_array_equal(frame, expected)
        with np.load(self.root / "scan/reference.npz") as reference:
            corrected = -np.log(np.clip((raw[1:5:2, 1:4:2, :].astype(np.float64) - 12) / 389, 1e-6, 1))
            np.testing.assert_array_equal(reference["sinogram"], corrected.transpose(1, 0, 2).astype(np.float32))
        args = parse_args(["--hdf5", str(self.file), "--sino", "1:4:2", "--proj", "1:5:2"])
        meta["reconstruction"] = reconstruction_options(args)
        workload = workload_description(args, meta)
        self.assertEqual(workload["detector_element"], "f32")
        self.assertEqual(workload["detector_frame_bytes"], 48)
        self.assertEqual(workload["corrected_frame_bytes"], 48)
        rates = streaming_rates(workload, meta, {"decompress": dict(published=4),
                                                "correct": dict(published=2)}, 2, 1)
        self.assertEqual(rates["stages"]["decompress"]["payload_mib_per_second"], 96 / 1024**2)
        self.assertEqual(rates["stages"]["correct"]["payload_mib_per_second"], 48 / 1024**2)

    def test_integer_calibration_stack_with_fractional_mean_uses_float32(self):
        with h5py.File(self.file, "a") as file:
            file["exchange/data_dark"][1] = 13
        meta = self.prepare()
        self.assertEqual(meta["element"], "f32")
        with np.load(self.root / "scan/reference.npz") as reference:
            np.testing.assert_array_equal(reference["dark"], np.full((4, 6), 11.5, np.float32))

    def test_counts_above_uint16_range_use_float32(self):
        with h5py.File(self.file, "a") as file:
            del file["exchange/data"]
            file["exchange/data"] = np.full((5, 4, 6), 70000, np.uint32)
            del file["exchange/data_white"]
            file["exchange/data_white"] = np.full((1, 4, 6), 100000, np.uint32)
        meta = self.prepare()
        self.assertEqual(meta["element"], "f32")
        with np.load(self.root / "scan/reference.npz") as reference:
            np.testing.assert_array_equal(reference["flat"], np.full((4, 6), 100000, np.float32))

    def test_invalid_data_is_rejected(self):
        cases = [
            ("exchange/data", np.full((5, 4, 6), 1e40), "counts must be"),
            ("exchange/data", np.full((5, 4, 6), -1), "counts must be"),
            ("exchange/data", np.full((5, 4, 6), np.nan), "finite detector counts"),
            ("exchange/data", np.zeros((4, 6)), "projections must have shape"),
            ("exchange/data_white", np.zeros((1, 4, 6)), "flat must exceed dark"),
            ("exchange/data_white", np.zeros((1, 3, 6)), "matching images"),
            ("exchange/data_dark", np.zeros((0, 4, 6)), "matching images"),
            ("exchange/data_dark", np.full((1, 4, 6), np.nan), "finite detector counts"),
            ("exchange/data_dark", np.full((4, 6), -0.1), "counts must be"),
            ("exchange/data_white", np.full((1, 4, 6), 1e40), "counts must be"),
            ("exchange/data_white", np.full((4, 6), np.inf), "finite detector counts"),
            ("exchange/data_white", np.full((1, 4, 6), 11.9), "flat must exceed dark"),
            ("exchange/theta", np.zeros(3), "one angle per input projection"),
            ("exchange/theta", np.full(5, np.inf), "finite angles"),
        ]
        for index, (path, values, error) in enumerate(cases):
            with self.subTest(path=path, error=error):
                bad_file = self.root / f"bad-{index}.h5"
                with h5py.File(self.file, "r") as source, h5py.File(bad_file, "w") as target:
                    source.copy("exchange", target)
                    del target[path]
                    target[path] = values
                with self.assertRaisesRegex(ValueError, error):
                    from_hdf5(self.root / f"bad-scan-{index}", bad_file)

    def test_missing_dataset_and_unknown_theta_units_fail(self):
        with self.assertRaisesRegex(ValueError, "missing HDF5 dataset"):
            self.prepare(flat_path="missing")
        with h5py.File(self.file, "a") as file:
            file["exchange/theta"].attrs["units"] = "turns"
        with self.assertRaisesRegex(ValueError, "unrecognized theta units"):
            self.prepare()

    def test_dimensions_and_launcher_defaults_come_from_selected_file(self):
        args = parse_args(["--hdf5", str(self.file), "--sino", "1:4:2", "--proj", "1:5:2",
                           "--live", "--stress", "--center", "3"])
        self.assertEqual((args.slices, args.pixels, args.angles), (2, 6, 2))
        self.assertEqual(args.algorithm, "gridrec")
        self.assertEqual((args.budget, args.scan_period), (8388608, 0))
        self.assertEqual(hdf5_dimensions(self.file), dict(rows=4, columns=6, angles=5))
        for flags in (["--pixels", "64"], ["--center", "7"], ["--sino", "4:4"], ["--proj", "0:6"]):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(["--hdf5", str(self.file), *flags])


class SelectionTests(unittest.TestCase):
    def test_selection_syntax_and_phantom_defaults(self):
        self.assertEqual(selection(":"), slice(None, None, 1))
        self.assertEqual(selection("2:10:3"), slice(2, 10, 3))
        for value in ("2", "1:2:3:4", "-1:3", "0:3:0", "0:3:-1", "a:b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                selection(value)
        args = parse_args([])
        self.assertEqual((args.pixels, args.slices, args.angles), (64, 8, 96))
        self.assertEqual(args.algorithm, "sirt")
        self.assertIsNone(args.hdf5)
        for flags in (["--sino", "0:2"], ["--display-max", "nan"], ["--display-max", "0"]):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)


if __name__ == "__main__":
    unittest.main()
