"""Independent NumPy generation and correction for the streamed calibration example."""
import numpy as np


def frame(shape, kind, calibration_id, sample, samples):
    pixel = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    dark = 100 + pixel % 31 + calibration_id * 3
    response = 1 + (pixel + calibration_id) % 4
    noise = 2 * sample - (samples - 1)
    if kind == 0:
        result = dark + noise
    elif kind == 1:
        illumination = np.where(pixel % 97 == 0, 0, np.where(pixel % 193 == 0, -16, response * 4096))
        result = dark + illumination + noise
    else:
        signal = (sample * 7 + pixel * 3 + calibration_id * 13) % 3072
        result = dark + response * signal
        result = np.where(pixel % 389 == 0, dark - 5, result)
        result = np.where(pixel % 251 == 0, 65535, result)
    return result.astype(np.uint16)


class Reference:
    def __init__(self, shape, meta, minimum):
        self.shape, self.meta, self.minimum = shape, meta, minimum
        self.maps = {}

    def calibration(self, calibration_id):
        if calibration_id not in self.maps:
            dark = np.mean([frame(self.shape, 0, calibration_id, i, self.meta["dark_frames"])
                            for i in range(self.meta["dark_frames"])], axis=0, dtype=np.float64)
            flat = np.mean([frame(self.shape, 1, calibration_id, i, self.meta["flat_frames"])
                            for i in range(self.meta["flat_frames"])], axis=0, dtype=np.float64)
            self.maps[calibration_id] = dark, flat
        return self.maps[calibration_id]

    def corrected(self, calibration_id, sample):
        dark, flat = self.calibration(calibration_id)
        raw = frame(self.shape, 2, calibration_id, sample, 1).astype(np.float64)
        valid = (flat - dark > self.minimum) & (raw != self.meta["saturation"])
        result = np.full(self.shape, np.nan, dtype=np.float64)
        np.divide(raw - dark, flat - dark, out=result, where=valid)
        return result
