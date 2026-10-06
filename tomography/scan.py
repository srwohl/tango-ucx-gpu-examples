"""Prepare a reproducible compressed detector scan and an independent correction reference."""
import json
from pathlib import Path

import lz4.block
import numpy as np


def generate(directory, pixels, slices, angles, gpu):
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
    projection_id, attenuation = astra.create_sino3d_gpu(
        phantom, projection_geometry, volume_geometry, gpuIndex=gpu)
    astra.data3d.delete(projection_id)
    row, column = np.indices((slices, pixels))
    dark = (16 + (row + column) % 7).astype(np.uint16)
    response = (4000 + (column % 5) * 100).astype(np.uint16)
    flat = dark + response
    raw = np.rint(dark[:, None, :] + response[:, None, :] * np.exp(-attenuation)).astype(np.uint16)
    return _write_scan(directory, theta, dark, flat,
                       (raw[:, i, :] for i in range(angles)),
                       dict(type="phantom"), phantom=phantom, raw=raw)


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
              theta_path="/exchange/theta", theta_units="auto", sino=None, proj=None):
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
                           source)


def _write_scan(directory, theta, dark, flat, projections, source, **reference):
    slices, pixels = dark.shape
    angles = len(theta)
    expected = np.empty((slices, angles, pixels), dtype=np.float32)
    response = flat.astype(np.float64) - dark
    frames = []
    with (directory / "compressed.bin").open("xb") as payloads:
        def inputs():
            yield 0, 0, 0.0, dark
            yield 1, 0, 0.0, flat
            for index, frame in enumerate(projections):
                transmission = (frame.astype(np.float64) - dark) / response
                expected[:, index, :] = -np.log(np.clip(transmission, 1e-6, 1))
                yield 2, index, float(theta[index]), frame

        for kind, projection, angle, frame in inputs():
            compressed = lz4.block.compress(np.ascontiguousarray(frame).tobytes(), store_size=False)
            frames.append(dict(kind=kind, projection=projection, theta=angle,
                               bytes=len(compressed), offset=payloads.tell()))
            payloads.write(compressed)
    meta = dict(example="gpu-tomography-v1", codec="lz4-raw",
                element="u16" if dark.dtype == np.uint16 else "f32",
                rows=slices, columns=pixels, angles=angles, theta=theta.tolist(),
                scan_id=1, calibration_id=1, source=source,
                max_compressed_bytes=max(frame["bytes"] for frame in frames), frames=frames)
    (directory / "scan.json").write_text(json.dumps(meta, indent=2))
    np.savez(directory / "reference.npz", sinogram=expected, dark=dark, flat=flat, **reference)
    return meta
