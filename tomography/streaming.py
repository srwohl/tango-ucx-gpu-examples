"""Sliding-window FBP: updates within a scan, and three arbitrary slices instead of a volume.

The reconstruction device keeps the latest rotation of projections in a ring and publishes
every few projections once the ring is full. In slice output the ring holds projections
filtered on arrival, and only the requested planes are backprojected, as RECAST3D and
TomoStream do. Nothing here depends on Tango.
"""
import numpy as np

from reconstruction import _fbp_filter

PLANES = 3


def update_interval(buffering, angles):
    """Projections between publications; 0 or absent keeps one publication per scan."""
    return buffering.get("update_projections", 0) or angles


def update_due(received, projection, angles, interval):
    """Publish after this projection? Never before the ring holds one rotation."""
    return received >= angles and (projection + 1) % interval == 0


def publications(completed_scans, angles, interval):
    """Publications of a run: the first scan fills the ring and publishes once."""
    return completed_scans and 1 + (completed_scans - 1) * (angles // interval)


def slice_size(rows, columns):
    """Edge of the square slice images: any axis-aligned plane of the volume fits."""
    return max(rows, columns)


def default_planes(rows, columns):
    """The three orthogonal planes through the volume centre: axial, coronal, sagittal.

    Each is centred in its square image on whole voxels, so nothing is interpolated.
    """
    size = slice_size(rows, columns)
    z, c = -((size - rows) // 2), -((size - columns) // 2)
    return [dict(origin=[rows // 2, c, c], u=[0, 1, 0], v=[0, 0, 1]),
            dict(origin=[z, columns // 2, c], u=[1, 0, 0], v=[0, 0, 1]),
            dict(origin=[z, c, columns // 2], u=[1, 0, 0], v=[0, 1, 0])]


def plane_array(planes):
    """Validate three planes as a (3, 3, 3) float64 array of origin, u and v in (z, y, x).

    Pixel (i, j) of a slice samples the volume at index coordinates origin + i * u + j * v.
    Points outside the volume are zero, so a plane can be placed, tilted and scaled freely.
    """
    if isinstance(planes, np.ndarray):
        result = np.array(planes, dtype=np.float64)
    else:
        if not isinstance(planes, (list, tuple)) or len(planes) != PLANES:
            raise ValueError(f"expected {PLANES} slice planes")
        rows = []
        for plane in planes:
            if not isinstance(plane, dict) or set(plane) != {"origin", "u", "v"}:
                raise ValueError("a slice plane has origin, u and v")
            for key in ("origin", "u", "v"):
                vector = plane[key]
                if (not isinstance(vector, (list, tuple)) or len(vector) != 3 or any(
                        isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector)):
                    raise ValueError(f"slice plane {key} must be three numbers (z, y, x)")
            rows.append([plane["origin"], plane["u"], plane["v"]])
        result = np.array(rows, dtype=np.float64)
    if result.shape != (PLANES, 3, 3) or not np.isfinite(result).all() or np.abs(result).max() > 1e6:
        raise ValueError("slice planes must be finite (z, y, x) coordinates")
    if not (np.linalg.norm(np.cross(result[:, 1], result[:, 2]), axis=1) > 1e-9).all():
        raise ValueError("slice plane u and v must span a plane")
    return result


def plane_list(planes):
    """The JSON form of a (3, 3, 3) plane array."""
    return [dict(origin=plane[0].tolist(), u=plane[1].tolist(), v=plane[2].tolist())
            for plane in np.asarray(planes, dtype=np.float64)]


_BACKPROJECT = r"""
extern "C" __global__
void backproject_planes(const float *filtered, const float *trig, const double *planes, float *out,
                        int rows, int angles, int columns, int count, int size, float scale) {
    const long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= (long long)count * size * size) return;
    const double *plane = planes + 9 * (n / ((long long)size * size));
    const int i = (n / size) % size, j = n % size;
    // Decide in double what lies inside the volume, as the host reference does.
    const double pz = plane[0] + i * plane[3] + j * plane[6];
    const double py = plane[1] + i * plane[4] + j * plane[7];
    const double px = plane[2] + i * plane[5] + j * plane[8];
    float value = 0;
    if(pz >= -0.5 && pz <= rows - 0.5 && py >= -0.5 && py <= columns - 0.5 &&
       px >= -0.5 && px <= columns - 0.5) {
        const float z = pz, y = py, x = px;
        // A slice of the volume is one detector row: interpolate between its neighbours.
        const float row = fminf(fmaxf(z, 0.0f), rows - 1.0f);
        const int z0 = min((int)row, max(rows - 2, 0)), z1 = min(z0 + 1, rows - 1);
        const float wz = row - z0, centre = 0.5f * (columns - 1);
        const float *lower = filtered + (size_t)z0 * angles * columns;
        const float *upper = filtered + (size_t)z1 * angles * columns;
        for(int a = 0; a < angles; ++a, lower += columns, upper += columns) {
            const float t = (x - centre) * trig[2 * a] + (y - centre) * trig[2 * a + 1] + centre;
            const float base = floorf(t), w = t - base;
            const int k = (int)base;
            // Rays that miss the detector contribute nothing.
            if(k >= 0 && k < columns) value += (1 - w) * (lower[k] + wz * (upper[k] - lower[k]));
            if(k >= -1 && k + 1 < columns) value += w * (lower[k + 1] + wz * (upper[k + 1] - lower[k + 1]));
        }
        value *= scale;
    }
    out[n] = value;
}
"""


class SliceReconstructor:
    """Filter projections into a ring as they arrive; backproject planes from it.

    The ring is the caller's (rows, angles, columns) float32 GPU array. Arrays passed to a
    call are used only during it, on the caller's current stream.
    """

    def __init__(self, shape, theta, cp):
        self.cp = cp
        self.rows, self.angles, self.columns = shape
        self.size = slice_size(self.rows, self.columns)
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.angles,):
            raise ValueError("angle count differs from the ring")
        self.trig = cp.asarray(np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32))
        self.planes = cp.empty((PLANES, 3, 3), dtype=cp.float64)
        # Without fused multiply-add, the kernel and the host place a pixel that lies exactly
        # on the volume's boundary on the same side of it.
        self.kernel = cp.RawKernel(_BACKPROJECT, "backproject_planes", options=("--fmad=false",))
        self.filter = None

    def configure(self, options):
        """The filter of projections that arrive from now on."""
        key = (options["filter"], options.get("filter_cutoff"))
        if key != self.filter:
            self.padded, response = _fbp_filter(self.columns, *key)
            self.response = self.cp.asarray(response)
            self.filter = key
        self.scale = np.float32(np.pi / (2 * self.angles) * options.get("scale_factor", 1.0))

    def store(self, ring, projection, frame):
        """Filter one (rows, columns) projection into its place in the ring."""
        cp = self.cp
        spectrum = cp.fft.rfft(frame, n=self.padded, axis=-1)
        spectrum *= self.response
        ring[:, projection, :] = cp.fft.irfft(spectrum, n=self.padded, axis=-1)[:, :self.columns]

    def backproject(self, ring, planes, output):
        """Fill the (3, size, size) output from a ring holding one rotation."""
        if ring.shape != (self.rows, self.angles, self.columns) or not ring.flags.c_contiguous:
            raise ValueError("ring differs from the scan geometry")
        if output.shape != (PLANES, self.size, self.size) or not output.flags.c_contiguous:
            raise ValueError("slice output differs from the scan geometry")
        self.planes.set(plane_array(planes))
        count = PLANES * self.size * self.size
        self.kernel(((count + 255) // 256,), (256,),
                    (ring, self.trig, self.planes, output, np.int32(self.rows), np.int32(self.angles),
                     np.int32(self.columns), np.int32(PLANES), np.int32(self.size), self.scale))


def reference_slices(sinogram, theta, options, planes):
    """The same planes on the host from an unfiltered (rows, angles, columns) sinogram.

    Written separately from the GPU kernel, in double precision; for verification only.
    """
    sinogram = np.asarray(sinogram, dtype=np.float32)
    rows, angles, columns = sinogram.shape
    padded, response = _fbp_filter(columns, options["filter"], options.get("filter_cutoff"))
    filtered = np.fft.irfft(np.fft.rfft(sinogram, n=padded, axis=-1) * response,
                            n=padded, axis=-1)[..., :columns]
    size, centre = slice_size(rows, columns), (columns - 1) / 2
    i, j = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    result = np.zeros((PLANES, size, size))
    for plane, image in zip(plane_array(planes), result):
        z, y, x = (plane[0][axis] + i * plane[1][axis] + j * plane[2][axis] for axis in range(3))
        inside = ((z >= -0.5) & (z <= rows - 0.5) & (y >= -0.5) & (y <= columns - 0.5) &
                  (x >= -0.5) & (x <= columns - 0.5))
        z, y, x = z[inside], y[inside], x[inside]
        row = np.clip(z, 0, rows - 1)
        z0 = np.minimum(row.astype(int), max(rows - 2, 0))
        z1 = np.minimum(z0 + 1, rows - 1)
        wz = row - z0
        total = np.zeros(len(z))
        for angle in range(angles):
            t = (x - centre) * np.cos(theta[angle]) + (y - centre) * np.sin(theta[angle]) + centre
            k = np.floor(t).astype(int)
            w = t - k
            for offset, weight in ((0, 1 - w), (1, w)):
                column = k + offset
                hit = (column >= 0) & (column < columns)
                column = np.clip(column, 0, columns - 1)
                lower, upper = filtered[z0, angle, column], filtered[z1, angle, column]
                total += np.where(hit, weight * (lower + wz * (upper - lower)), 0)
        image[inside] = total
    result *= np.pi / (2 * angles) * options.get("scale_factor", 1.0)
    return result.astype(np.float32)


class Updates:
    """One device's publications in order: (array copy, record) with identity checks.

    Several publications share a scan; the last projection of the window tells them apart.
    """

    def __init__(self, sub, scan_id, angles, interval):
        self.sub, self.scan_id, self.angles, self.interval = sub, scan_id, angles, interval
        self.last = None

    def __getattr__(self, name):
        return getattr(self.sub, name)

    def read(self, timeout=0.05):
        batch = self.sub.read(timeout=timeout)
        if batch is None:
            return None
        if batch.frames != 1:
            raise ValueError("expected one reconstruction publication per batch")
        record = batch.records[0]
        identity = (int(record["scan_id"]), int(record["projection"]))
        if (identity[0] < self.scan_id or not 0 <= identity[1] < self.angles or
                (identity[1] + 1) % self.interval or (self.last is not None and identity <= self.last)):
            raise ValueError("reconstruction update identity did not advance")
        self.last = identity
        return batch.array[0].copy(), record.copy()

    def complete(self, record):
        """Does this publication end on a scan boundary, holding exactly that scan?"""
        return int(record["projection"]) + 1 == self.angles
