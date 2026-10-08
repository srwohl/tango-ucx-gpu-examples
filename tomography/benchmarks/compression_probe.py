"""nvCOMP GPU compression rate and ratio for uint16 detector frames, against the detector rate.

Synthetic frames are a sphere with Poisson noise. Pass --frames with real detector frames:
the ratio depends on the data far more than on the GPU.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--frames", type=Path, help=".npy of uint16 frames, (count, rows, columns)")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument("--count", type=int, default=16, help="synthetic frames per batch")
    parser.add_argument("--counts", type=float, default=20000, help="synthetic flat-field counts")
    parser.add_argument("--fps", type=float, default=240, help="detector frame rate")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import cupy as cp
    from nvidia import nvcomp

    cp.cuda.Device(args.gpu).use()
    if args.frames:
        frames = cp.asarray(np.load(args.frames))
        if frames.dtype != cp.uint16 or frames.ndim != 3:
            parser.error("--frames needs a uint16 array of (count, rows, columns)")
        source = str(args.frames)
    else:
        y, x = cp.mgrid[0:args.rows, 0:args.columns].astype(cp.float32)
        radius = 0.44 * min(args.rows, args.columns)
        r = cp.sqrt((y - args.rows / 2) ** 2 + (x - args.columns / 2) ** 2) / radius
        transmission = cp.where(r < 1, cp.exp(-2 * cp.sqrt(cp.maximum(0, 1 - r**2))), 1)
        frames = cp.random.poisson(cp.broadcast_to(
            args.counts * transmission, (args.count, args.rows, args.columns))).astype(cp.uint16)
        source = f"synthetic sphere, {args.counts:g} counts, Poisson noise"
    count = len(frames)
    typed = [nvcomp.as_array(frames[i].reshape(-1)) for i in range(count)]
    as_bytes = [nvcomp.as_array(frames[i].view(cp.uint8).reshape(-1)) for i in range(count)]
    needed = frames[0].nbytes * args.fps / 2**30
    cases = [("lz4 raw, one chunk per frame", dict(algorithm="LZ4", bitstream_kind=nvcomp.BitstreamKind.RAW), as_bytes),
             ("lz4", dict(algorithm="LZ4"), as_bytes),
             ("lz4 uint16", dict(algorithm="LZ4", data_type="<u2"), typed),
             ("gdeflate", dict(algorithm="GDeflate"), as_bytes),
             ("zstd", dict(algorithm="Zstd"), as_bytes),
             ("cascaded uint16", dict(algorithm="Cascaded", data_type="<u2"), typed),
             ("bitcomp uint16", dict(algorithm="Bitcomp", data_type="<u2"), typed)]
    results = []
    for label, options, arrays in cases:
        try:
            codec = nvcomp.Codec(device_id=args.gpu, **options)
            rates = []
            for repeat in range(args.repeats + 1):
                cp.cuda.runtime.deviceSynchronize()
                start = time.perf_counter()
                encoded = codec.encode(arrays)
                size = sum(item.buffer_size for item in encoded)
                cp.cuda.runtime.deviceSynchronize()
                if repeat:
                    rates.append(frames.nbytes / (time.perf_counter() - start) / 2**30)
            results.append(dict(codec=label, gib_per_second=float(np.median(rates)),
                                ratio=frames.nbytes / size,
                                keeps_pace_with_detector=float(np.median(rates)) > needed,
                                samples_gib_per_second=rates))
        except Exception as error:  # a codec or option this nvCOMP build lacks
            results.append(dict(codec=label, error=f"{type(error).__name__}: {error}"))
    result = dict(gpu=cp.cuda.runtime.getDeviceProperties(args.gpu)["name"].decode(),
                  nvcomp=nvcomp.__version__, frames=source, shape=list(frames.shape),
                  detector_gib_per_second=needed, results=results,
                  scope="Compression only; no decompression, storage or transport")
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
