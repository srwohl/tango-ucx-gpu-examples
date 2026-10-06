"""GPU reconstruction, explicit CPU GridRec, and independent host references."""
from contextlib import ExitStack

import numpy as np

ALGORITHMS = ("sirt", "fbp", "gridrec")
FILTERS = ("ram-lak", "shepp-logan", "hann", "parzen")
GRIDREC_FILTERS = {"ram-lak": "ramlak", "shepp-logan": "shepp", "hann": "hann", "parzen": "parzen"}


def live_configuration(options, columns):
    """Validate a complete JSON settings object, including canonical reports.

    Null fields belonging to other methods in canonical reports are ignored.
    An update replaces the complete configuration; omitted knobs use defaults.
    """
    if not isinstance(options, dict):
        raise ValueError("reconstruction settings must be an object")
    allowed = {"algorithm", "backend", "filter", "iterations", "threads", "relaxation",
               "min_constraint", "max_constraint", "filter_cutoff", "center",
               "gaussian_fwhm", "scale_factor", "slices_per_block"}
    if options.keys() - allowed:
        raise ValueError(f"unknown reconstruction settings: {sorted(options.keys() - allowed)}")
    method = options.get("algorithm")
    if method not in ALGORITHMS:
        raise ValueError("algorithm must be sirt, fbp or gridrec")
    required_numeric = {"gaussian_fwhm", "scale_factor", "slices_per_block"}
    required_numeric.update({"sirt": ("iterations", "relaxation"),
                             "gridrec": ("threads",), "fbp": ()}[method])
    for key in required_numeric:
        if key in options and options[key] is None:
            raise ValueError(f"{key} must be numeric")
    if "filter" in options and options["filter"] not in FILTERS:
        if options["filter"] is not None or method != "sirt":
            raise ValueError("unknown reconstruction filter")
    values = {key: value for key, value in options.items()
              if key not in ("algorithm", "backend", "filter") and value is not None}
    if method == "sirt":
        for key in ("min_constraint", "max_constraint"):
            if key in options:
                values[key] = options[key]
    for key, value in values.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{key} must be numeric or null")
    result = configuration(method, options.get("filter") or "ram-lak", **values)
    if result["center"] is not None and result["center"] > columns:
        raise ValueError("center must lie within the detector [0, columns]")
    if "backend" in options and options["backend"] != result["backend"]:
        raise ValueError("backend is selected by the algorithm")
    return result


def configuration(algorithm, filter_name="ram-lak", iterations=40, threads=4, *,
                  relaxation=1.0, min_constraint=0.0, max_constraint=None,
                  filter_cutoff=None, center=None, gaussian_fwhm=0.0, scale_factor=1.0,
                  slices_per_block=0):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown reconstruction algorithm: {algorithm}")
    if filter_name not in FILTERS:
        raise ValueError(f"unknown reconstruction filter: {filter_name}")
    for name, value in (("iterations", iterations), ("threads", threads)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (isinstance(slices_per_block, bool) or not isinstance(slices_per_block, (int, np.integer))
            or slices_per_block < 0):
        raise ValueError("slices_per_block must be a nonnegative integer (0 means all slices)")
    _sirt_parameters(relaxation, min_constraint, max_constraint)
    if algorithm != "sirt" and (relaxation != 1 or min_constraint != 0 or max_constraint is not None):
        raise ValueError("relaxation and min/max constraints apply only to SIRT")
    if filter_cutoff is not None:
        if not np.isfinite(filter_cutoff) or not 0 < filter_cutoff <= 1:
            raise ValueError("filter cutoff must be finite and in (0, 1]")
        if algorithm != "fbp" or filter_name not in ("shepp-logan", "hann"):
            raise ValueError("filter cutoff requires FBP with shepp-logan or hann")
    if center is not None:
        if not np.isfinite(center) or center < 0:
            raise ValueError("center must be a finite nonnegative detector coordinate")
        if algorithm != "gridrec":
            raise ValueError("center is currently supported only by GridRec")
    if not np.isfinite(gaussian_fwhm) or gaussian_fwhm < 0:
        raise ValueError("Gaussian FWHM must be finite and nonnegative")
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale factor must be finite and positive")
    return dict(algorithm=algorithm, backend="tomopy-cpu" if algorithm == "gridrec" else "astra-cuda",
                filter=None if algorithm == "sirt" else filter_name,
                iterations=iterations if algorithm == "sirt" else None,
                threads=threads if algorithm == "gridrec" else None,
                relaxation=relaxation if algorithm == "sirt" else None,
                min_constraint=min_constraint if algorithm == "sirt" else None,
                max_constraint=max_constraint if algorithm == "sirt" else None,
                filter_cutoff=filter_cutoff, center=center,
                gaussian_fwhm=gaussian_fwhm, scale_factor=scale_factor,
                slices_per_block=int(slices_per_block))


def slice_blocks(rows, slices_per_block=0):
    """Ordered, contiguous z ranges; zero retains the whole-volume backend call."""
    if rows < 1:
        raise ValueError("slice count must be positive")
    if (isinstance(slices_per_block, bool) or not isinstance(slices_per_block, (int, np.integer))
            or slices_per_block < 0):
        raise ValueError("slices_per_block must be a nonnegative integer (0 means all slices)")
    size = min(slices_per_block or rows, rows)
    for start in range(0, rows, size):
        yield slice(start, min(start + size, rows))


def _sirt_parameters(relaxation, minimum, maximum):
    if not np.isfinite(relaxation) or not 0 < relaxation < 2:
        raise ValueError("SIRT relaxation must be finite and in (0, 2)")
    for value in (minimum, maximum):
        if value is not None and not np.isfinite(value):
            raise ValueError("SIRT constraints must be finite or None")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum constraint exceeds maximum constraint")


def postprocess(volume, options, *, gpu=False):
    """In-place 3D Gaussian (voxel FWHM, reflect boundaries), then scaling.

    The GPU path uses only device arrays; defaults do no work or allocation.
    Constraints apply during SIRT, before these optional output transforms.
    """
    fwhm = options.get("gaussian_fwhm", 0.0)
    if fwhm:
        if gpu:
            from cupyx.scipy.ndimage import gaussian_filter
        else:
            from scipy.ndimage import gaussian_filter
        gaussian_filter(volume, fwhm / np.sqrt(8 * np.log(2)), output=volume, mode="reflect")
    scale = options.get("scale_factor", 1.0)
    if scale != 1:
        volume *= np.float32(scale)
    return volume


def _fbp_options(gpu, filter_name, filter_cutoff):
    configuration("fbp", filter_name, filter_cutoff=filter_cutoff)
    options = {"GPUindex": gpu, "FilterType": filter_name}
    if filter_cutoff is not None:
        options["FilterD"] = filter_cutoff
    return options


def gridrec(sinogram, theta, filter_name, threads, *, center=None):
    """TomoPy GridRec, with padding and the pipeline's z/y/x volume convention."""
    import tomopy

    columns = sinogram.shape[-1]
    configuration("gridrec", filter_name, threads=threads, center=center)
    if center is not None and center > columns:
        raise ValueError("center must lie within the detector [0, columns]")
    grid_size = columns + columns % 2
    # Explicit padding avoids Fourier wraparound, including power-of-two detectors.
    pad = (columns + 1) // 2
    padded = np.pad(sinogram, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    result = tomopy.recon(padded, theta, center=(columns / 2 if center is None else center) + pad,
                         sinogram_order=True, algorithm="gridrec", ncore=threads,
                         filter_name=GRIDREC_FILTERS[filter_name],
                         num_gridx=grid_size, num_gridy=grid_size)
    # Match parallel3d volume rows: TomoPy's reconstructed row order is reversed.
    result = result[:, ::-1, :]
    # Its FFT grid is centered on even dimensions; crop odd detector widths.
    result = result[:, :columns, grid_size - columns:]
    return np.ascontiguousarray(result, dtype=np.float32)


def _gpu_arrays(sinogram, output, theta, gpu):
    """Reject inputs that could make ASTRA stage through host or another device."""
    import cupy as cp

    for name, array in (("sinogram", sinogram), ("output", output)):
        if not isinstance(array, cp.ndarray):
            raise TypeError(f"{name} must be a CuPy GPU array; host staging is disabled")
        if array.dtype != cp.float32 or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous float32; implicit copies are disabled")
        if array.device.id != gpu:
            raise ValueError(f"{name} must be on GPU {gpu}; cross-device copies are disabled")
    if sinogram.ndim != 3 or any(size < 1 for size in sinogram.shape):
        raise ValueError("sinogram must have nonempty (rows, angles, columns) dimensions")
    rows, angles, columns = sinogram.shape
    if output.shape != (rows, columns, columns) or len(theta) != angles:
        raise ValueError("output shape or angle count differs from the sinogram")


class SirtGPU:
    """Reusable SIRT scratch and weights for one fixed scan geometry.

    Borrowed input and publisher output arrays are used only during ``run``.
    No transport pointer or ASTRA data object survives the call.
    """

    def __init__(self, shape, theta, gpu):
        import astra
        import cupy as cp

        self.shape, self.gpu = tuple(shape), gpu
        rows, angles, columns = self.shape
        self.theta = np.array(theta, dtype=np.float32, copy=True)
        if min(self.shape) < 1 or self.theta.shape != (angles,):
            raise ValueError("SIRT shape or angle count is invalid")
        self.projection_geometry = astra.create_proj_geom(
            "parallel3d", 1.0, 1.0, rows, columns, self.theta)
        self.volume_geometry = astra.create_vol_geom(columns, columns, rows)
        with cp.cuda.Device(gpu):
            self.ray_weights = cp.empty(self.shape, dtype=cp.float32)
            self.voxel_weights = cp.empty((rows, columns, columns), dtype=cp.float32)
            self.residual = cp.empty_like(self.ray_weights)
            self.update = cp.empty_like(self.voxel_weights)
        # Normalize in one kernel, without a temporary boolean-index array.
        self.normalize = cp.ElementwiseKernel(
            "float32 value", "float32 weight",
            "weight = value < 1e-6f ? 0.0f : 1.0f / value;", "tomography_sirt_weights")
        self.weights_ready = False

    def run(self, sinogram, output, iterations, *, relaxation=1.0,
            min_constraint=0.0, max_constraint=None):
        import astra
        import cupy as cp

        _gpu_arrays(sinogram, output, self.theta, self.gpu)
        configuration("sirt", iterations=iterations, relaxation=relaxation,
                      min_constraint=min_constraint, max_constraint=max_constraint)
        if sinogram.shape != self.shape or iterations < 1:
            raise ValueError("SIRT sinogram shape differs or iterations are not positive")
        with cp.cuda.Device(self.gpu), ExitStack() as cleanup:
            projector = astra.projector3d.create(dict(
                type="cuda3d", ProjectionGeometry=self.projection_geometry,
                VolumeGeometry=self.volume_geometry, option={"GPUindex": self.gpu}))
            cleanup.callback(astra.projector3d.delete, projector)

            def project(operation, source, destination):
                # ASTRA owns its streams: explicitly order each CuPy/ASTRA handoff.
                cp.cuda.get_current_stream().synchronize()
                try:
                    operation(projector, source, out=destination)
                finally:
                    cp.cuda.runtime.deviceSynchronize()

            try:
                if not self.weights_ready:
                    output.fill(1)
                    self.residual.fill(1)
                    project(astra.projector3d.direct_FP, output, self.ray_weights)
                    project(astra.projector3d.direct_BP, self.residual, self.voxel_weights)
                    for weights in (self.ray_weights, self.voxel_weights):
                        self.normalize(weights, weights)
                    cp.cuda.get_current_stream().synchronize()
                    self.weights_ready = True
                output.fill(0)
                for _ in range(iterations):
                    project(astra.projector3d.direct_FP, output, self.residual)
                    cp.subtract(sinogram, self.residual, out=self.residual)
                    self.residual *= self.ray_weights
                    project(astra.projector3d.direct_BP, self.residual, self.update)
                    self.update *= self.voxel_weights
                    if relaxation != 1:
                        self.update *= np.float32(relaxation)
                    output += self.update
                    if min_constraint is not None:
                        cp.maximum(output, min_constraint, out=output)
                    if max_constraint is not None:
                        cp.minimum(output, max_constraint, out=output)
            finally:
                # Complete writes before returning the borrowed output slot, also on error.
                cp.cuda.get_current_stream().synchronize()


def fbp_gpu(sinogram, output, theta, gpu, filter_name, *, filter_cutoff=None):
    """DLPack GPU links; ASTRA 2.5 additionally uses internal device-to-device scratch."""
    import astra
    import cupy as cp

    _gpu_arrays(sinogram, output, theta, gpu)
    backend_options = _fbp_options(gpu, filter_name, filter_cutoff)
    columns = sinogram.shape[-1]
    # ASTRA's 2D and parallel3d angle conventions have opposite signs.
    pg = astra.create_proj_geom("parallel", 1.0, columns, -theta)
    vg = astra.create_vol_geom(columns, columns)
    with cp.cuda.Device(gpu):
        cp.cuda.get_current_stream().synchronize()
        for row in range(sinogram.shape[0]):
            with ExitStack() as cleanup:
                projections = astra.data2d.link("-sino", pg, sinogram[row])
                cleanup.callback(astra.data2d.delete, projections)
                volume = astra.data2d.link("-vol", vg, output[row])
                cleanup.callback(astra.data2d.delete, volume)
                config = astra.astra_dict("FBP_CUDA")
                config.update(ProjectionDataId=projections, ReconstructionDataId=volume,
                              option=backend_options)
                algorithm = astra.algorithm.create(config)
                cleanup.callback(astra.algorithm.delete, algorithm)
                try:
                    astra.algorithm.run(algorithm)
                finally:
                    # ASTRA owns its streams; complete writes before unlinking borrowed arrays.
                    cp.cuda.runtime.deviceSynchronize()


def reference_reconstruction(sinogram, theta, gpu, options, *, block_rows=None):
    """Independent host-corrected input and ASTRA objects, outside timed acquisition."""
    if block_rows is None:
        result = _reference_reconstruction(sinogram, theta, gpu, options)
    else:
        result = np.empty((sinogram.shape[0], sinogram.shape[2], sinogram.shape[2]), np.float32)
        for block in slice_blocks(sinogram.shape[0], block_rows):
            result[block] = _reference_reconstruction(sinogram[block], theta, gpu, options)
    return postprocess(result, options)


def _reference_reconstruction(sinogram, theta, gpu, options):
    method = options["algorithm"]
    if method == "gridrec":
        return gridrec(sinogram, theta, options["filter"], options["threads"],
                       center=options.get("center"))
    import astra

    rows, _, columns = sinogram.shape
    if method == "fbp":
        pg = astra.create_proj_geom("parallel", 1.0, columns, -theta)
        vg = astra.create_vol_geom(columns, columns)
        result = np.empty((rows, columns, columns), dtype=np.float32)
        for row in range(rows):
            with ExitStack() as cleanup:
                projections = astra.data2d.create("-sino", pg, sinogram[row])
                cleanup.callback(astra.data2d.delete, projections)
                volume = astra.data2d.create("-vol", vg)
                cleanup.callback(astra.data2d.delete, volume)
                config = astra.astra_dict("FBP_CUDA")
                config.update(ProjectionDataId=projections, ReconstructionDataId=volume,
                              option=_fbp_options(gpu, options["filter"], options.get("filter_cutoff")))
                algorithm = astra.algorithm.create(config)
                cleanup.callback(astra.algorithm.delete, algorithm)
                astra.algorithm.run(algorithm)
                result[row] = astra.data2d.get(volume)
        return result
    if method != "sirt":
        raise ValueError(f"unknown reconstruction algorithm: {method}")
    pg = astra.create_proj_geom("parallel3d", 1.0, 1.0, rows, columns, theta)
    vg = astra.create_vol_geom(columns, columns, rows)
    with ExitStack() as cleanup:
        projections = astra.data3d.create("-sino", pg, sinogram)
        cleanup.callback(astra.data3d.delete, projections)
        volume = astra.data3d.create("-vol", vg)
        cleanup.callback(astra.data3d.delete, volume)
        config = astra.astra_dict("SIRT3D_CUDA")
        relaxation = options.get("relaxation", 1.0)
        minimum, maximum = options.get("min_constraint", 0.0), options.get("max_constraint")
        backend_options = {"GPUindex": gpu}
        # ASTRA has no public SIRT3D relaxation option. For nonunit relaxation,
        # take one unconstrained ASTRA step, blend on the host, then constrain.
        if relaxation == 1:
            if minimum is not None:
                backend_options["MinConstraint"] = minimum
            if maximum is not None:
                backend_options["MaxConstraint"] = maximum
        config.update(ProjectionDataId=projections, ReconstructionDataId=volume,
                      option=backend_options)
        algorithm = astra.algorithm.create(config)
        cleanup.callback(astra.algorithm.delete, algorithm)
        if relaxation == 1:
            astra.algorithm.run(algorithm, options["iterations"])
        else:
            previous = np.zeros((rows, columns, columns), dtype=np.float32)
            for _ in range(options["iterations"]):
                astra.algorithm.run(algorithm, 1)
                current = astra.data3d.get(volume)
                current -= previous
                current *= np.float32(relaxation)
                current += previous
                if minimum is not None:
                    np.maximum(current, minimum, out=current)
                if maximum is not None:
                    np.minimum(current, maximum, out=current)
                astra.data3d.store(volume, current)
                previous = current
        return astra.data3d.get(volume)
