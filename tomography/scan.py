"""Prepare a reproducible compressed detector scan and an independent correction reference."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil

import lz4.block
import numpy as np

# Generated scans kept for reuse; the least recently used are removed beyond this.
CACHE_BYTES = 8 * 1024**3


def orb_mask(pixels, slices):
    """A visible ellipsoid, using broadcast coordinates rather than a coordinate volume."""
    horizontal = (np.arange(pixels, dtype=np.float32) + .5) / pixels * 2 - 1
    vertical = (np.arange(slices, dtype=np.float32) + .5) / slices * 2 - 1
    return (((horizontal[None, None, :] - .35) / .16)**2 +
            (horizontal[None, :, None] / .23)**2 + (vertical[:, None, None] / .8)**2 <= 1)


def orb_phase(cycle, period):
    """The orb dims and brightens alike, so a period holds period // 2 + 1 different scans."""
    cycle %= period
    return min(cycle, period - cycle)


def orb_scale(cycle, period):
    return np.float32(.6 + .4 * np.cos(2 * np.pi * orb_phase(cycle, period) / period))


def detector_counts(attenuation, dark, response):
    transmission = np.negative(attenuation)
    np.exp(transmission, out=transmission)
    transmission *= response
    transmission += dark
    np.rint(transmission, out=transmission)
    return transmission.astype(np.uint16)


class DynamicPhantom:
    """The stored projections of every orb brightness: playback reads them and computes nothing."""

    def __init__(self, directory, scan):
        directory, source = Path(directory), scan["source"]
        self.period, self.angles = source["orb_period"], scan["angles"]
        self.compressed = scan["codec"] != "raw"
        self.offsets = np.fromfile(directory / source["phase_index"], "<u8")
        self.payloads = np.memmap(directory / source["phase_file"], np.uint8, "r")
        with np.load(directory / "reference.npz") as reference:
            self.dark = reference["dark"]
            self.flat = reference["flat"]
        self.response = (self.flat - self.dark).astype(np.float32)

    def payload(self, cycle, projection):
        """One projection of a scan, as the source publishes it."""
        index = orb_phase(cycle, self.period) * self.angles + projection
        return self.payloads[self.offsets[index]:self.offsets[index + 1]].tobytes()

    def frame(self, cycle, projection):
        payload = self.payload(cycle, projection)
        if self.compressed:
            payload = lz4.block.decompress(payload, uncompressed_size=self.dark.nbytes)
        return np.frombuffer(payload, np.uint16).reshape(self.dark.shape)

    def sinogram(self, cycle):
        expected = np.empty((self.dark.shape[0], self.angles, self.dark.shape[1]), np.float32)
        for projection in range(self.angles):
            transmission = (self.frame(cycle, projection).astype(np.float64) - self.dark) / self.response
            expected[:, projection, :] = -np.log(np.clip(transmission, 1e-6, 1))
        return expected


def uncompressed_scan(directory, scan):
    """Prepare fixed-size detector payloads for a pipeline without a decoder."""
    directory = Path(directory)
    frame_bytes = scan["rows"] * scan["columns"] * (4 if scan["element"] == "f32" else 2)
    frames = []
    with (directory / "compressed.bin").open("rb") as source, (directory / "raw.bin").open("xb") as output:
        for frame in scan["frames"]:
            payload = lz4.block.decompress(source.read(frame["bytes"]), uncompressed_size=frame_bytes)
            if len(payload) != frame_bytes:
                raise ValueError("detector frame size differs from scan geometry")
            frames.append(dict(frame, bytes=frame_bytes, offset=output.tell()))
            output.write(payload)
    return dict(scan, codec="raw", frames=frames, payload_file="raw.bin")


def _write_phases(directory, attenuation, orb, dark, response, period, codec):
    """Store the projections of each orb brightness in the detector codec, and where each begins.

    The source then replays a changing phantom without arithmetic. A raw 360 x 360 detector
    with 720 projections and a period of 40 takes 3.7 GiB.
    """
    angles = attenuation.shape[1]

    def payload(phase, projection):
        frame = detector_counts(attenuation[:, projection, :] +
                                orb_scale(phase, period) * orb[:, projection, :], dark, response)
        raw = frame.tobytes()
        return lz4.block.compress(raw, store_size=False) if codec == "lz4-raw" else raw

    offsets = [0]
    # Measured: beyond four threads the interpreter lock gives back what the cores gain.
    with (directory / "phases.bin").open("xb") as payloads, ThreadPoolExecutor(4) as pool:
        for phase in range(period // 2 + 1):
            # One brightness at a time bounds the payloads waiting to be written.
            for data in pool.map(lambda projection: payload(phase, projection), range(angles)):
                payloads.write(data)
                offsets.append(offsets[-1] + len(data))
    np.asarray(offsets, "<u8").tofile(directory / "phases.idx")
    return dict(phase_file="phases.bin", phase_index="phases.idx")


def _stored_bytes(directory):
    return sum(file.stat().st_size for file in directory.iterdir())


def generate(directory, pixels, slices, angles, gpu, orb_period=0, *, codec="lz4-raw", cache=None):
    """Write a phantom scan. With ``cache``, a scan generated before is linked from there instead."""
    if cache is None:
        return _generate(directory, pixels, slices, angles, gpu, orb_period, codec)
    # This file's digest retires the scans of another version of the generator.
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    stored = Path(cache) / f"phantom-{pixels}x{slices}x{angles}-orb{orb_period}-{codec}-{digest}"
    if not stored.is_dir():
        stored.parent.mkdir(parents=True, exist_ok=True)
        building = stored.with_name(f".{stored.name}.{os.getpid()}")
        try:
            _generate(building, pixels, slices, angles, gpu, orb_period, codec)
            try:
                building.rename(stored)
            except OSError:
                if not stored.is_dir():  # Otherwise another run stored the same scan first.
                    raise
        finally:
            shutil.rmtree(building, ignore_errors=True)
    os.utime(stored)
    others = sorted((path for path in stored.parent.glob("phantom-*") if path != stored),
                    key=lambda path: path.stat().st_mtime, reverse=True)
    total = _stored_bytes(stored)
    for path in others:
        total += _stored_bytes(path)
        if total > CACHE_BYTES:
            shutil.rmtree(path, ignore_errors=True)
    directory = Path(directory)
    directory.mkdir()
    for file in stored.iterdir():
        if file.name == "scan.json":
            shutil.copy(file, directory)  # The launcher adds its settings to this one.
            continue
        try:
            os.link(file, directory / file.name)
        except OSError:  # Another filesystem.
            os.symlink(file.resolve(), directory / file.name)
    return json.loads((directory / "scan.json").read_text())


def _generate(directory, pixels, slices, angles, gpu, orb_period, codec):
    import astra

    if not astra.use_cuda():
        raise RuntimeError("install the CUDA build of ASTRA; this example requires a GPU")
    directory = Path(directory)
    directory.mkdir()
    theta = np.linspace(0, np.pi, angles, endpoint=False, dtype=np.float32)
    volume_geometry = astra.create_vol_geom(pixels, pixels, slices)
    projection_geometry = astra.create_proj_geom("parallel3d", 1.0, 1.0, slices, pixels, theta)
    phantom_id, phantom = astra.data3d.shepp_logan(volume_geometry)
    astra.data3d.delete(phantom_id)
    phantom *= np.float32(0.01)
    source = dict(type="phantom")
    if orb_period:
        mask = orb_mask(pixels, slices)
        orb = mask.astype(np.float32) * np.float32(.01)
        phantom[mask] = 0
        orb_id, orb_attenuation = astra.create_sino3d_gpu(
            orb, projection_geometry, volume_geometry, gpuIndex=gpu)
        astra.data3d.delete(orb_id)
        del orb
        source["orb_period"] = orb_period
    projection_id, attenuation = astra.create_sino3d_gpu(
        phantom, projection_geometry, volume_geometry, gpuIndex=gpu)
    astra.data3d.delete(projection_id)
    if orb_period:
        phantom[mask] = np.float32(.01)
    row, column = np.indices((slices, pixels))
    dark = (16 + (row + column) % 7).astype(np.uint16)
    response = (4000 + (column % 5) * 100).astype(np.uint16)
    flat = dark + response
    if orb_period:
        projections = (detector_counts(attenuation[:, index, :] + orb_attenuation[:, index, :],
                                       dark, response) for index in range(angles))
        meta = _write_scan(directory, theta, dark, flat, projections, source, codec=codec, phantom=phantom)
    else:
        raw = np.rint(dark[:, None, :] + response[:, None, :] * np.exp(-attenuation)).astype(np.uint16)
        meta = _write_scan(directory, theta, dark, flat,
                           (raw[:, index, :] for index in range(angles)), source, codec=codec, phantom=phantom, raw=raw)
    if orb_period:
        frame_bytes = slices * pixels * 2
        meta["max_compressed_bytes"] = max(meta["max_compressed_bytes"], frame_bytes + frame_bytes // 255 + 16)
        meta["source"].update(_write_phases(directory, attenuation, orb_attenuation, dark, response,
                                            orb_period, codec))
        (directory / "scan.json").write_text(json.dumps(meta, indent=2))
    return meta


def selection(value):
    """Parse a nonnegative, forward START:STOP[:STEP] selection."""
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("selection must be START:STOP[:STEP]")
    try:
        start, stop = (int(part) if part else None for part in parts[:2])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    except ValueError as error:
        raise ValueError("selection must contain integers") from error
    if (start is not None and start < 0) or (stop is not None and stop < 0) or step < 1:
        raise ValueError("selection bounds must be nonnegative and step must be positive")
    return slice(start, stop, step)


def _selected(value, size, name):
    value = value or slice(None)
    start = 0 if value.start is None else value.start
    stop = size if value.stop is None else value.stop
    step = 1 if value.step is None else value.step
    if not 0 <= start < stop <= size or step < 1:
        raise ValueError(f"{name} selection must be nonempty and within [0, {size}]")
    return slice(start, stop, step), len(range(start, stop, step))


def _h5py():
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("HDF5 input requires h5py; install it in the examples environment") from error
    return h5py


def _dataset(file, path):
    h5py = _h5py()
    if path not in file or not isinstance(file[path], h5py.Dataset):
        raise ValueError(f"missing HDF5 dataset: {path}")
    return file[path]


def _layout(file, data_path, sino, proj):
    data = _dataset(file, data_path)
    if data.ndim != 3 or min(data.shape) < 1:
        raise ValueError("projections must have shape (angles, detector rows, detector columns)")
    rows, row_count = _selected(sino, data.shape[1], "row")
    projections, angle_count = _selected(proj, data.shape[0], "projection")
    return data, rows, projections, dict(rows=row_count, columns=data.shape[2], angles=angle_count)


def hdf5_dimensions(filename, *, data_path="/exchange/data", sino=None, proj=None):
    with _h5py().File(filename, "r") as file:
        return _layout(file, data_path, sino, proj)[3]


def _validated_counts(values, name):
    if values.dtype.kind not in "uif" or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite detector counts")
    if np.any(values < 0) or np.any(values > np.finfo(np.float32).max):
        raise ValueError(f"{name} counts must be nonnegative and within the float32 range")
    return values


def _counts(values, name, dtype):
    values = _validated_counts(values, name)
    if dtype == np.uint16 and (np.any(values > 65535) or np.any(values != np.rint(values))):
        raise ValueError(f"{name} counts do not fit uint16 without loss")
    return np.ascontiguousarray(values, dtype=dtype)


def _calibration(file, path, detector_shape, rows):
    data = _dataset(file, path)
    if data.ndim == 2 and data.shape == detector_shape:
        return _validated_counts(data[rows, :], path).astype(np.float64)
    if data.ndim != 3 or data.shape[0] < 1 or data.shape[1:] != detector_shape:
        raise ValueError(f"{path} must be a detector image or a nonempty stack of matching images")
    # Keep only one calibration image in memory at a time. Preserve fractional
    # means; choose the detector stream dtype after both calibrations are read.
    total = np.zeros((len(range(*rows.indices(detector_shape[0]))), detector_shape[1]), np.float64)
    for index in range(data.shape[0]):
        total += _validated_counts(data[index, rows, :], path)
    return total / data.shape[0]


def from_hdf5(directory, filename, *, data_path="/exchange/data",
              flat_path="/exchange/data_white", dark_path="/exchange/data_dark",
              theta_path="/exchange/theta", theta_units="auto", sino=None, proj=None, codec="lz4-raw"):
    """Prepare raw Data Exchange HDF5 data as uint16 or float32 detector frames."""
    if theta_units not in ("auto", "degrees", "radians"):
        raise ValueError("theta units must be auto, degrees or radians")
    with _h5py().File(filename, "r") as file:
        data, rows, projections, _ = _layout(file, data_path, sino, proj)
        dark = _calibration(file, dark_path, data.shape[1:], rows)
        flat = _calibration(file, flat_path, data.shape[1:], rows)
        integral_calibration = all(np.all(values <= 65535) and np.all(values == np.rint(values))
                                   for values in (dark, flat))
        dtype = (np.uint16 if data.dtype.kind in "ui" and data.dtype.itemsize <= 2 and
                 integral_calibration else np.float32)
        dark, flat = _counts(dark, dark_path, dtype), _counts(flat, flat_path, dtype)
        if np.any(flat <= dark):
            raise ValueError("flat must exceed dark at every selected detector pixel")
        generated_theta = theta_path == "/exchange/theta" and theta_path not in file
        units = theta_units
        if generated_theta:
            # Match the APS/DXchange fallback, then select the original indices.
            theta = np.linspace(0, np.pi, data.shape[0], dtype=np.float64)[projections]
            units = "radians"
        else:
            dataset = _dataset(file, theta_path)
            if dataset.shape != (data.shape[0],):
                raise ValueError("theta must be a 1D array with one angle per input projection")
            theta = np.asarray(dataset[projections], dtype=np.float64)
            if units == "auto":
                attribute = dataset.attrs.get("units", "degrees")
                if isinstance(attribute, bytes):
                    attribute = attribute.decode("utf-8")
                attribute = str(attribute).strip().lower()
                if attribute in ("degree", "degrees", "deg"):
                    units = "degrees"
                elif attribute in ("radian", "radians", "rad"):
                    units = "radians"
                else:
                    raise ValueError("unrecognized theta units; set --theta-units degrees or radians")
            if units == "degrees":
                theta = np.deg2rad(theta)
        theta = theta.astype(np.float32)
        if not np.isfinite(theta).all():
            raise ValueError("theta must contain finite angles")
        source = dict(type="hdf5", file=str(Path(filename).resolve()), data_path=data_path,
                      flat_path=flat_path, dark_path=dark_path, theta_path=theta_path,
                      theta_units=units, generated_theta=generated_theta,
                      sino=[rows.start, rows.stop, rows.step],
                      proj=[projections.start, projections.stop, projections.step],
                      calibration="mean in float64, stored as " + np.dtype(dtype).name)
        directory = Path(directory)
        directory.mkdir()
        return _write_scan(directory, theta, dark, flat,
                           (_counts(data[index, rows, :], data_path, dtype)
                            for index in range(projections.start, projections.stop, projections.step)),
                           source, codec=codec)


def _write_scan(directory, theta, dark, flat, projections, source, *, codec="lz4-raw", **reference):
    if codec not in ("lz4-raw", "raw"):
        raise ValueError("detector codec must be lz4-raw or raw")
    slices, pixels = dark.shape
    angles = len(theta)
    expected = np.empty((slices, angles, pixels), dtype=np.float32)
    response = flat.astype(np.float64) - dark
    frames = []
    payload_file = "compressed.bin" if codec == "lz4-raw" else "detector.bin"
    with (directory / payload_file).open("xb") as payloads:
        def inputs():
            yield 0, 0, 0.0, dark
            yield 1, 0, 0.0, flat
            for index, frame in enumerate(projections):
                transmission = (frame.astype(np.float64) - dark) / response
                expected[:, index, :] = -np.log(np.clip(transmission, 1e-6, 1))
                yield 2, index, float(theta[index]), frame

        for kind, projection, angle, frame in inputs():
            raw_bytes = np.ascontiguousarray(frame).tobytes()
            compressed = lz4.block.compress(raw_bytes, store_size=False) if codec == "lz4-raw" else raw_bytes
            frames.append(dict(kind=kind, projection=projection, theta=angle,
                               bytes=len(compressed), offset=payloads.tell()))
            payloads.write(compressed)
    meta = dict(example="gpu-tomography-v1", codec=codec, payload_file=payload_file,
                element="u16" if dark.dtype == np.uint16 else "f32",
                rows=slices, columns=pixels, angles=angles, theta=theta.tolist(),
                scan_id=1, calibration_id=1, source=source,
                max_compressed_bytes=max(frame["bytes"] for frame in frames), frames=frames)
    (directory / "scan.json").write_text(json.dumps(meta, indent=2))
    np.savez(directory / "reference.npz", sinogram=expected, dark=dark, flat=flat, **reference)
    return meta
