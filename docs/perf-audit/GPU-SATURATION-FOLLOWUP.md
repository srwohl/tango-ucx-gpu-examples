# Why the GPU does not saturate under MPS

Measured 2026-10-08 on the local RTX 2060 SUPER with the MPS daemon running, ASTRA 2.5.0
and CuPy 14.2. Single runs, clocks not locked, desktop and browser sharing the GPU.
No production code changed.

Run in question:

```sh
pixi run tomography-live --gpu-stress --algorithm fbp --network auto \
  --transport-batch 16 --reconstructors 1 --pixels 256 --slices 256 --view-history-mib 1024 \
  --no-saving --no-decompression --no-verify-volumes --update-projections 48
```

Geometry: 256 slices, 720 projections, 256 columns; a volume every 48 projections.

## The reconstruct stage is the limit, and it is bound by one host thread

Counters from `status.json` of that live run at 166 s, 502 volumes published:

| Stage | Time waiting for output room | Time inside `reconstruct()` |
| --- | ---: | ---: |
| source | 143.5 s (86%) | |
| correct | 160.4 s (97%) | |
| reconstruct | 0.0012 s | 159.5 s (96%) |

- Reconstruct spends 96% of the wall inside `fbp_gpu`: **318 ms per volume, 1.24 ms per slice**.
- Its worker thread sat at 93% of one core (`top -H`) while `nvidia-smi` showed 54 to 60%.
- It never waited for an output slot, so the viewer is not holding it back.
- Source and correct were idle, waiting for room, 86% and 97% of the time.

`fbp_gpu` backprojects one slice at a time: copy, stream sync, `astra.algorithm.run`,
device sync, copy, 256 times per volume. The GPU is idle between those calls while
Python and ASTRA's wrapper run on the host. This is finding 2 in `NOTES.md`, now the
whole wall time because the sliding window reconstructs 15 volumes per scan.

MPS cannot fill those gaps. It lets kernels from different processes overlap; here one
process does nearly all the GPU work, serially, with a blocking sync after every slice.

## `nvidia-smi` overstates what the pipeline uses

With the pipeline stopped and the browser still open, the same counter read 24 to 34%.
It reports the share of time any kernel ran, so the two do not add, but the 57% seen
live is not all pipeline work.

## More reconstructors is not available here, and did not help where it is

`--update-projections` with `--reconstructors 2` is rejected: "sliding-window updates and
slice output use one reconstructor in one chain".

Whole-scan volumes (`--update-projections 0`, 60 scans, no viewer), MPS on:

| Reconstructors | Volumes/s | Frames/s | Upstream wait for room | GPU, mean of mid-run samples |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.97 | 2146 | 1.4 s source, 13.4 s correct | 51% |
| 2 | 3.25 | 2341 | 0.01 s, 0.02 s | 52% |
| 3 | 2.55 | 1838 | 0.04 s, 0.06 s | 54% |

With two reconstructors the upstream waits vanish and throughput rises 9%: the limit moves
to the per-frame path before reconstruction, at roughly 2300 frames/s. These counters do
not separate source from correct. Three is slower than two.

## Same pipeline without the viewer

The user's options, 12 scans, no `--live`: **4.23 volumes/s**, 226 ms per volume, 95% of the
wall in `reconstruct()`, GPU 66%. The live run gave 3.0 to 3.2 volumes/s. The viewer and
browser were not isolated from each other, and the two runs differ in length.

## What the reconstruction costs without the per-slice wrapper

`probes/fbp_native.py` at this geometry, pipeline stopped, median of 5:

| Method | 256 slices | Against production |
| --- | ---: | ---: |
| Production `fbp_gpu` | 210 ms | 1.00x |
| Direct native BP | 138 ms | 1.52x |
| Persistent native BP | 133 ms | 1.58x |
| Custom slice-batch BP | 108 ms | 1.94x |

All four match the independent ASTRA reference to 5e-8. The isolated ratio is an upper
bound for the pipeline; the earlier paired run turned 1.72x isolated into 1.17x.

## Slice output already avoids the loop

`--output-mode slices` (40 scans, no viewer, otherwise the same options) backprojects three
planes with one `RawKernel` launch per update from a ring filtered on arrival
(`streaming.SliceReconstructor`):

| Output | Updates/s | Frames/s | Time inside reconstruction | GPU, mean of mid-run samples |
| --- | ---: | ---: | ---: | ---: |
| Volume, 256 slices | 4.23 | 220 | 95% of the wall | 66% |
| Three planes | 38.8 | 1905 | 1.05 s of 15.1 s (7%) | 38% |

Neither source nor correct waits for room in slice mode, so the limit is again the per-frame
path at about 1900 frames/s. The GPU reading is close to the idle desktop baseline.

## TomocuPy at this geometry

`probes/tomocupy_fbp.py` (TomocuPy d14ece0, built by `probes/tomocupy/build.sh`), 256 slices,
720 angles, 256 columns, Ram-Lak, median of 5, pipeline stopped. "Host call" is the time until
the call returns; the rest is the GPU finishing.

| Method | Per volume | Host call | Rel. L2 to production | Rel. L2 to phantom |
| --- | ---: | ---: | ---: | ---: |
| Production `fbp_gpu` | 208 ms | 208 ms | 0 | 0.0546 |
| Persistent native ASTRA BP | 139 ms | 139 ms | 0 | 0.0546 |
| TomocuPy LpRec, chunk 32 | 90 ms | 2.0 ms | 0.021 | 0.0567 |
| TomocuPy LineRec, chunk 128 | 185 ms | 0.3 ms | 0.016 | 0.0578 |
| TomocuPy FourierRec, chunk 128 | 210 ms | 0.8 ms | 0.027 | 0.0582 |

- TomocuPy removes the host cost almost entirely; the GPU then sets the rate.
- Only LpRec is faster overall here. LineRec is direct backprojection, 1.12x.
- None matches ASTRA within the pipeline's verification tolerance (2 to 3% of voxels do),
  and the output needs a 180 degree turn, a flip, a half-pixel shift and a 0.78 scale to line up.
- LpRec and FourierRec leave the corners outside the inscribed circle at zero.
- At 2048 columns and 2000 angles (`tomocupy-fbp-target.json`, 8 slices): LpRec 4.3x and
  FourierRec 3.1x faster than production, LineRec 2.1x slower.

## TomocuPy in the pipeline (2026-10-09)

`fourierrec`, `lprec` and `linerec` are now `--algorithm` choices
(`tomography/tomocupy_backend.py`, built by `pixi run build-tomocupy`). Same options as the
run in question, 12 scans, no viewer, MPS on, single runs:

| Algorithm | Volumes/s | Against `fbp` | Per volume in `reconstruct()` | GPU, median of mid-run samples |
| --- | ---: | ---: | ---: | ---: |
| `fbp` | 4.17 | 1.00x | 229 ms | 67% |
| `lprec` | 8.79 | 2.11x | 103 ms | 99% |
| `linerec` | 4.59 | 1.10x | 208 ms | 100% |
| `fourierrec` | 4.23 | 1.01x | 226 ms | 100% |

- The GPU is now the limit for all three. Only `lprec` turns that into throughput at this size.
- Filtering through CuPy's FFT cost 40 ms per volume; TomocuPy's in-place filter, given
  (angles, slices, padded) scratch so its per-slice weights become per-angle, costs 23 ms
  with identical output. That took `lprec` from 7.61 to 8.79 volumes/s.
- Volumes are in the pipeline's convention. TomocuPy's grid is centred on pixel `columns // 2`,
  its rows run the other way and its values are 4 / pi larger; one phase ramp per angle in the
  filter weights, negated angles (a reversed sinogram for `lprec`) and a scale fix all three.
  `linerec` at an odd width then matches ASTRA FBP to 1e-4, which isolates the convention.
- Against ASTRA FBP at 256 columns and 360 angles: `linerec` 1.4%, `lprec` 2.2%,
  `fourierrec` 2.7% relative L2 inside the circle.

Found on the way:

- `fourierrec` transforms two slices as one complex image and each takes part of its
  partner: 0.7% relative L2 at 64 columns, 0.2% at 256, 0.1% at 512, with upstream's own
  angles and no shift as well. Its volume depends on the block layout.
- `fourierrec` in `float16` repeats itself only to 6e-5 on a 0.028 peak; in `float32` to 6e-9.
- `linerec` writes zeros for a one-slice chunk: it interpolates between neighbouring rows.
- `fourierrec` returns zeros for an odd detector width. `lprec` exits the process on angles
  it cannot use; the backend checks them first.
- `tomography/stream_smoke.py` fails before it switches algorithms: it expects one byte per
  voxel from `/api/volume` and the viewer now serves `uint16-le-zyx`. A copy with that one
  check relaxed switched through all six algorithms on the same devices.

Not measured: full-size detectors in the pipeline, several GPUs, and the live page in a real
browser (`puppeteer-core` is not installed; the fake-DOM page tests pass).

## Reproduce

```sh
OUT=/tmp/gpu-saturation docs/perf-audit/probes/gpu_saturation_run.sh r1 --reconstructors 1
OUT=/tmp/gpu-saturation docs/perf-audit/probes/gpu_saturation_run.sh w2 \
  --reconstructors 2 --update-projections 0 --scans 60
pixi run python docs/perf-audit/probes/fbp_native.py --rows 256 --columns 256 --angles 720 \
  --filters ram-lak --repeats 5 --output /tmp/fbp-native-live-geometry.json
pixi run build-tomocupy
OUT=/tmp/gpu-saturation docs/perf-audit/probes/gpu_saturation_run.sh lprec \
  --reconstructors 1 --algorithm lprec
```

Evidence: `gpu-saturation/` (run summaries, GPU samples, probe output). The live run's
`status.json` was in a temporary directory and is not retained; its numbers are above.
