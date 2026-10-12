"""TomocuPy's FourierRec, LpRec and LineRec in the pipeline's volume convention.

TomocuPy (https://github.com/tomography/tomocupy, Argonne National Laboratory) backprojects a
block of slices in a few kernel launches. ``build_tomocupy.sh`` builds its reconstruction
modules; its file readers and command line configuration are not used.

Its grid is centred on pixel ``columns // 2`` where ASTRA's is centred on ``(columns - 1) / 2``,
its rows run the other way and its values are 4 / pi times larger. All three are folded into the
filtering here, so every algorithm's volume can be displayed and compared alike.
"""
import sys

import numpy as np

from reconstruction import (FBP_FILTER_SAMPLES, TOMOCUPY_ALGORITHMS, TOMOCUPY_DTYPES,
                            TOMOCUPY_FILTERS, _gpu_arrays, slice_blocks, tomocupy_build,
                            tomocupy_geometry)


def _modules():
    build = str(tomocupy_build())
    if build not in sys.path:
        # Ahead of any installed TomocuPy: its package __init__ imports the whole application.
        sys.path.insert(0, build)
    try:
        from tomocupy.reconstruction import fbp_filter, fourierrec, linerec, lprec
    except ImportError as error:
        raise RuntimeError("the TomocuPy methods are not built: run `pixi run build-tomocupy`") from error
    return fbp_filter, fourierrec, linerec, lprec


def padded_size(columns, dtype="float32"):
    """TomocuPy's padded detector length; half precision cuFFT needs a power of two."""
    size = 4 * columns
    return 1 << (size - 1).bit_length() if dtype == "float16" else size


def chunk_rows(algorithm, rows, angles, columns, dtype="float32"):
    """Slices per TomocuPy call, bounded like FBP's filter scratch."""
    chunk = max(1, min(rows, FBP_FILTER_SAMPLES // (angles * padded_size(columns, dtype))))
    if algorithm == "fourierrec":
        # Two slices are transformed as one complex image.
        return chunk + chunk % 2
    # LineRec interpolates between neighbouring detector rows and writes nothing for a single one.
    return max(chunk, 2) if algorithm == "linerec" else chunk


def scratch_bytes(algorithm, rows, angles, columns, dtype="float32"):
    """Device memory this module owns for one engine; TomocuPy's plans and grids come on top."""
    chunk, size = chunk_rows(algorithm, rows, angles, columns, dtype), padded_size(columns, dtype)
    item = np.dtype(dtype).itemsize
    # The padded copy and its spectrum, the cropped result, the weights and a volume block.
    return (2 * chunk * angles * size * item + chunk * angles * columns * item +
            angles * (size // 2 + 1) * 2 * item + chunk * columns * columns * item)


class TomocupyGPU:
    """Filter scratch and TomocuPy objects for one geometry, reused across volumes.

    Borrowed input and output arrays are used only during ``run``, on the caller's stream.
    """

    def __init__(self, algorithm, theta, columns, rows, gpu, filter_name="ram-lak", *,
                 center=None, dtype="float32"):
        import cupy as cp

        if algorithm not in TOMOCUPY_ALGORITHMS or filter_name not in TOMOCUPY_FILTERS:
            raise ValueError("unknown TomocuPy algorithm or filter")
        if dtype not in TOMOCUPY_DTYPES:
            raise ValueError("dtype must be float32 or float16")
        self.theta = np.array(theta, dtype=np.float32, copy=True)
        angles = len(self.theta)
        if rows < 1 or angles < 1:
            raise ValueError("TomocuPy shape or angle count is invalid")
        center = columns / 2 if center is None else center
        if not np.isfinite(center) or not 0 <= center <= columns:
            raise ValueError("center must lie within the detector [0, columns]")
        tomocupy_geometry(dict(algorithm=algorithm, dtype=dtype), columns, self.theta)
        fbp_filter, fourierrec, linerec, lprec = _modules()
        self.algorithm, self.gpu, self.columns, self.dtype = algorithm, gpu, columns, dtype
        self.rows = chunk_rows(algorithm, rows, angles, columns, dtype)
        self.size = size = padded_size(columns, dtype)
        self.pad = size // 2 - columns // 2
        # The rotation axis as a detector index; the detector middle is (columns - 1) / 2.
        axis = center - 0.5
        if algorithm == "lprec":
            # LpRec takes no angles: the mirrored object is the same sinogram with its angles
            # and detector reversed. It expects the axis in the detector middle.
            angle = np.arange(angles) * np.pi / angles
            shift, grid = (columns - 1) / 2 - (columns - 1 - axis), 0.5
        else:
            # Negated angles mirror the rows. These grids put the axis on pixel columns // 2.
            angle = -self.theta.astype(np.float64)
            shift, grid = columns // 2 - axis, columns // 2 - (columns - 1) / 2
        with cp.cuda.Device(gpu):
            # TomocuPy's filter gives each leading index its own weights: slices upstream, angles
            # here, so the padded scratch is (angles, slices, size).
            self.filter = fbp_filter.FBPFilter(size, self.rows, angles, dtype)
            response = cp.asnumpy(self.filter.calc_filter(TOMOCUPY_FILTERS[filter_name])).astype(np.float64)
            # One phase ramp per angle moves the axis onto TomocuPy's and the object half a pixel back.
            move = shift - grid * (np.cos(angle) + np.sin(angle))
            phase = np.exp(-2j * np.pi * move[:, None] * np.fft.rfftfreq(size)[None, :])
            weights = (response[None, :] * phase * (np.pi / 4)).astype(np.complex64)
            self.weights = cp.asarray(np.ascontiguousarray(weights.view(np.float32).astype(dtype)))
            self.padded = cp.zeros((angles, self.rows, size), dtype=dtype)
            self.filtered = cp.zeros((self.rows, angles, columns), dtype=dtype)
            given = cp.asarray(angle, dtype=cp.float32)
            try:
                if algorithm == "fourierrec":
                    self.backprojector = fourierrec.FourierRec(columns, angles, self.rows, given, dtype)
                elif algorithm == "linerec":
                    self.backprojector = linerec.LineRec(given, angles, angles, self.rows, self.rows,
                                                         columns, dtype)
                else:
                    self.backprojector = lprec.LpRec(columns, angles, self.rows, given, dtype)
            except SystemExit as error:
                # Upstream exits the process on angles it cannot use.
                raise ValueError(f"TomocuPy {algorithm} rejected the scan geometry") from error

    def run(self, sinogram, output):
        import cupy as cp

        _gpu_arrays(sinogram, output, self.theta, self.gpu)
        if sinogram.shape[2] != self.columns:
            raise ValueError("sinogram width differs from the TomocuPy geometry")
        pad, columns = self.pad, self.columns
        with cp.cuda.Device(self.gpu):
            stream = cp.cuda.get_current_stream()
            try:
                for block in slice_blocks(sinogram.shape[0], self.rows):
                    count = block.stop - block.start
                    data, padded = sinogram[block], self.padded[:, :count]
                    if self.algorithm == "lprec":
                        padded[0, :, pad:pad + columns] = data[:, 0]
                        padded[1:, :, pad:pad + columns] = data[:, :0:-1, ::-1].transpose(1, 0, 2)
                    else:
                        padded[:, :, pad:pad + columns] = data.transpose(1, 0, 2)
                    # TomocuPy extends the detector with its edge values before filtering.
                    padded[:, :, :pad] = padded[:, :, pad:pad + 1]
                    padded[:, :, pad + columns:] = padded[:, :, pad + columns - 1:pad + columns]
                    # In place, on every row of the scratch: a shorter block leaves the rest stale.
                    self.filter.fslv.filter(self.padded.data.ptr, self.weights.data.ptr, stream.ptr)
                    self.filtered[:count] = padded[:, :, pad:pad + columns].transpose(1, 0, 2)
                    if count == self.rows and self.dtype == "float32":
                        self.backprojector.backprojection(output[block], self.filtered, stream)
                        continue
                    # A shorter last block or half precision goes through owned scratch.
                    if not hasattr(self, "volume"):
                        self.volume = cp.empty((self.rows, columns, columns), dtype=self.dtype)
                    self.filtered[count:] = 0
                    self.backprojector.backprojection(self.volume, self.filtered, stream)
                    output[block] = self.volume[:count]
            finally:
                # Complete writes before returning the borrowed output slot, also on error.
                stream.synchronize()


_REFERENCE = {}


def reference(sinogram, theta, gpu, options):
    """A host sinogram through the same backend: TomocuPy has no separate host implementation.

    This checks what reached the reconstruction device, not the method itself. It follows the
    device's blocks: FourierRec transforms slices in pairs, and each takes about 0.2% of its
    partner at 256 columns, so another pairing gives another volume.
    """
    import cupy as cp

    sinogram = np.ascontiguousarray(sinogram, dtype=np.float32)
    rows, _, columns = sinogram.shape
    block_rows = min(options.get("slices_per_block") or rows, rows)
    key = (options["algorithm"], np.asarray(theta, dtype=np.float32).tobytes(), columns, block_rows, gpu,
           options["filter"], options.get("center"), options.get("dtype") or "float32")
    with cp.cuda.Device(gpu):
        if key not in _REFERENCE:
            # One engine at a time: a new setting replaces the previous one's device memory.
            _REFERENCE.clear()
            _REFERENCE[key] = TomocupyGPU(key[0], theta, columns, block_rows, gpu, key[5], center=key[6],
                                          dtype=key[7])
        source = cp.asarray(sinogram)
        output = cp.empty((rows, columns, columns), dtype=cp.float32)
        for block in slice_blocks(rows, block_rows):
            _REFERENCE[key].run(source[block], output[block])
        return output.get()
