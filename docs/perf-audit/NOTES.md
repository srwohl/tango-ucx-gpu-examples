# Tomography pipeline performance audit — trail

Started 2026-10-07. Scope: `tango-ucx-gpu-examples/tomography/processors.py` first, then the
transport underneath it. Nothing in either repo has been changed; every result below came from
probes in `docs/perf-audit/probes/` and from running the unmodified demo.

Machine: one RTX 2060 SUPER (8 GB), CUDA 12.9, CuPy 14.2, ASTRA 2.5.0, UCX 1.22, Nsight Systems
2026.5.1. All pipeline stages share that one GPU without MPS, so absolute numbers will not carry
over to the cluster. The ratios and the mechanisms should.

## Where things stand

| # | Finding | Status |
|---|---|---|
| 1 | SIRT has no handoff penalty: 92% of wall time is ASTRA kernels | measured |
| 2 | FBP spends about 2/3 of its time outside backprojection kernels | measured |
| 3 | Layout and DLPack are correct on both boundaries; ASTRA refuses anything else | measured |
| 4 | GPU memory leaving for a host (or over TCP) moves in 8 KB fragments | measured, cause identified |
| 5 | Raising `UCX_SYSV_SEG_SIZE` breaks a subscription with "invalid frame" | reproduced 2/2, not diagnosed |
| 6 | Decompress costs about 1.1 ms/frame on its own; batching it moves the bottleneck downstream | measured |
| 7 | The publisher loop spins on `cudaEventQuery`; it is not what slows decompress | hypothesis refuted |
| 8 | nsys `--trace=ucx` cannot be used with anything that loads Tango | reproduced, no workaround |
| 9 | One startup failure ("decompress did not become ready") | seen once, not reproduced |
| 10 | The demo's own volume verification halves benchmark throughput and hid everything behind it | measured; corrects 4 and 6 |
| 11 | Volume writer now receives on the GPU: fragmenting gone, gain about 5% | change made, uncommitted |
| 12 | Batch limit raised from 16 to 512; decompress stops scaling at 16 | change made, uncommitted; measured |
| 13 | At the target geometry (2048 x 2048, 2000 angles) FBP needs 69 ms per slice here: 141 s per volume against an 8.75 s scan | measured locally; L40 not measured |

Suggested order for what comes next is at the end.

## How to reproduce

All commands run from `tango-ucx-gpu-examples/`.

```bash
NSYS=/opt/nvidia/nsight-systems/2026.5.1/target-linux-x64/nsys
P=/home/swohl/repo/tango-hpc/tango-ucx-gpu-examples/docs/perf-audit/probes

# Reconstruction handoffs, no transport (finding 1, 2)
$NSYS profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o baseline \
    pixi run python $P/handoff_probe.py

# What ASTRA asks of DLPack, what it rejects, layout in all four modes (finding 3)
pixi run python $P/dlpack_probe.py
pixi run python $P/astra_readonly.py

# Real tango-ucx GPU receive view (finding 3) - run from tango-ucx/
PYTHONPATH=build-gpu/python:python/tests TANGO_UCX_FIXTURE_DEVICE=build-gpu/tests/fixture_device \
    pixi run -e gpu python -m pytest -q -p no:cacheprovider $P/test_transport_dlpack.py -s

# Full pipeline with NVTX labels on Processor.consume etc. (finding 4, 6, 7)
PYTHONPATH=$P/shim $NSYS profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o pipeline-auto \
    pixi run python tomography/demo.py --pixels 512 --slices 128 --angles 360 \
    --algorithm fbp --scans 3 --scan-period 0 --network auto --output /tmp/run-auto

# Which UCX protocol carries each link (finding 4)
UCX_PROTO_INFO=y pixi run python tomography/demo.py <same arguments> --output /tmp/proto
#   then read /tmp/proto/{source,decompress,correct,reconstruct}.log

# Decompress stage alone (finding 6, 7)
pixi run python $P/decompress_alone.py
```

The shim (`probes/shim/sitecustomize.py`) wraps `Processor.consume`, `consume_many`,
`reconstruct` and `reconstruct_block` in NVTX ranges at import time. It works inside the embedded
Python of `pipeline_device` because the demo passes `PYTHONPATH` through.

Report analysis used the Nsight skill at
`/opt/nvidia/nsight-systems/2026.5.1/skills/nsight-systems/` (`scripts/nsys_skill_cli.py
report-fact` and `report-query`). Process IDs in a pipeline trace map to roles through the run's
`devices.json`. Traces from this session are in `docs/perf-audit/traces/`.

## 1. SIRT: nothing to win in the handoffs

Probe: `handoff_probe.py`, 128 x 360 x 512, 32-row blocks, 20 iterations, host-buffered path.

- 6.77 s per volume. 96% of it is inside `direct_FP` / `direct_BP`, 92% is ASTRA kernel time.
- 866 `cudaDeviceSynchronize` calls cost 3.6 ms in total. The "device-wide handoff" comment at
  `processors.py:155` is accurate but the cost is nil.
- ASTRA creates a private stream per call and synchronises it before returning. No ASTRA kernel
  ran on the compute stream or the default stream. `with self.stream:` cannot move ASTRA.
- The block prefetch upload (2.3 ms for 24 MB) never overlapped an ASTRA kernel, and it does not
  matter against 1.7 s of compute per block.
- Each direct call copies its input into a CUDA 3D array (ASTRA internal, about 0.2 ms) and
  writes its output in place. Fixed cost per call is 1.2 to 1.4 ms, most of it one
  `cudaGetDeviceProperties` at about 0.9 ms.

Conclusion: SIRT speed is ASTRA's projector speed. Only fewer iterations or a different
projector would change it.

## 2. FBP: the overhead is per slice

Same probe. 0.19 s per 128-slice volume, 1.5 ms wall per slice against 0.46 ms of `devBP`.

- Inside each `algorithm.run` ASTRA allocates two pitched buffers, copies the linked sinogram
  and image in and out, and frees them.
- Our two copies per slice (`filtered[...] = ...`, `output[row] = image`) cost microseconds of
  device time. Relinking per slice instead would cost 0.47 ms for the link pair alone, so the
  scratch design in `fbp_gpu` is the cheaper one.
- Removing the overhead means one block backprojection, which the `fbp_gpu` docstring rejected
  because `parallel3d` interpolates differently, beyond the verification tolerance.

## 3. Layout and DLPack

Two boundaries, two exporters:

| Boundary | Exporter | Result |
|---|---|---|
| tango-ucx receive view -> CuPy | tango-ucx `GpuView.__dlpack__` | zero-copy, C-contiguous, 256-byte aligned, CuPy passes its current stream so the batch hold is ordered on it |
| CuPy -> ASTRA | CuPy | zero-copy link, C-contiguous float32 in all four modes (gpu/host sinogram x volume/block output) |

- Sinogram is (rows, angles, columns), volume is (z, y, x). Checked by lighting one z-slice and
  forward projecting.
- ASTRA raises `ValueError` for strided, Fortran-order, float64 or wrong-shape arrays. It never
  copies silently. `_gpu_arrays` (`reconstruction.py:170`) checks first anyway.
- ASTRA passes only `max_version=(1, 0)` to `__dlpack__`, no stream. So the explicit
  `stream.synchronize()` before each ASTRA call (`reconstruction.py:236`) is required.
- ASTRA rejects read-only DLPack tensors even as input, and tango-ucx exports are always
  read-only. A tango-ucx view can never be handed to ASTRA directly. It would be the wrong
  grouping anyway: (frames, rows, columns).
- `pipeline_device.cpp` does not use the Python DLPack path at all. It passes raw pointers into
  `array_at`. `owner=None` there is fine: no borrowed array outlives its call, and C++ records
  the slot event on the same stream at publish (`publisher.cpp:443`).

Not checked: a real publisher slot from the running C++ device (a CuPy allocation stood in).

## 4. GPU memory to a host goes out in 8 KB pieces

Pipeline: 512 x 128 x 360, FBP, 1086 frames, all stages on GPU 0.

| | `--network tcp` | `--network auto` |
|---|---|---|
| Links | tcp/lo | sysv + cuda_ipc |
| First-to-last frame at reconstruct | 6.1 s | 2.2 s |
| `cuStreamSynchronize` time in `correct` | 5.1 s | negligible |

`UCX_PROTO_INFO=y` says, for `--network auto`:

- GPU to GPU: `zero-copy flushed write to remote` over `cuda_ipc`. One device copy per frame.
- Host to GPU (source to decompress): `fragmented copy-in copy-out` over `sysv`.
- GPU to host (reconstruct to the volume writer): `fragmented copy-in copy-out` over `sysv`.

The fragment is `UCX_SYSV_SEG_SIZE=8256`. In the trace, three 134 MB volumes left the GPU as
48,867 copies with a median of 8,240 bytes, 1.1 s of `cuMemcpyAsync` API time. Under TCP every
GPU frame is fragmented the same way (`UCX_TCP_TX_SEG_SIZE=8K`), each fragment followed by a
blocking stream synchronise: about 50 pairs per frame in `correct`, 79 in `reconstruct`.

In the 10-scan runs the reconstruct stage waited 1.4 to 2.7 s (of about 11 s) for a publisher
slot. I first attributed that to the fragmenting. That was wrong: see finding 10. The writer was
slow because it verifies every volume, and the fragmenting itself costs little.

## 5. Raising the segment size breaks the transport

`UCX_SYSV_SEG_SIZE=256K` and `=1M` both failed, 2 of 2, with the archive subscription reporting
`'failure': 'the publisher sent an invalid frame'` after 84 frames (`subscription.cpp:102`).
The check that fails is in `on_frame` (`subscription.cpp:73`). I did not find which clause.
With a larger segment UCX moves the 75 KB compressed frames from multi-fragment to single
copy-in, so something in the opaque-ring validation appears to depend on how UCX delivers
them. Until this is understood, finding 4 cannot be fixed by an environment variable.

Note: `TANGO_UCX_UCX_SYSV_SEG_SIZE` is reported as unused by UCX; the plain `UCX_` name is what
took effect.

## 6. Decompress

- In the pipeline with cuda_ipc, decompress is inside `consume` for 97% of its window: median
  1.04 ms per frame.
- Alone, with no transport and no other process (`decompress_alone.py`): 1.06 to 1.13 ms per
  frame. So the cost is intrinsic to the stage.
- In the pipeline trace the per-frame cost is: LZ4 kernel about 0.5 ms, the kernel launch call
  0.26 ms median (a launch is normally a few microseconds), five `cudaEventRecord` calls
  averaging 0.09 ms each.
- One frame is one LZ4 chunk, so the GPU decodes it serially. That is what the existing
  `--processing-mode batched` is for.

10 scans, `--network auto`, unprofiled:

| Mode | Elapsed | Frames/s | Who waits |
|---|---|---|---|
| scalar | 11.15 s | 325 | source 3.2 s, correct 3.5 s |
| `--transport-batch 16` | 11.09 s | 326 | same |
| + `--processing-batch 16 --processing-mode batched` | 10.28 s | 352 | decompress 2.3 s, correct 4.3 s, reconstruct 2.7 s |

These three rows were taken with volume verification on, which caps the run (finding 10). With
`--no-verify-volumes`, two runs each:

| Mode | Elapsed | Frames/s |
|---|---|---|
| scalar | 5.95 s, 6.72 s | 608, 539 |
| batched 16 | 4.08 s, 3.84 s | 888, 943 |

So batched decompress is worth about 1.5x, not 8%. In batched mode the remaining waits are
source 1.8 s, decompress 1.5 s and correct 2.1 s (summed over the run), with reconstruct at 0.

Open: why the LZ4 launch call takes 0.26 ms. nsys captured no CUDA activity for
`decompress_alone.py` in two attempts (NVTX only), so I could not compare launch latency with
and without the pipeline around it.

## 7. Event polling: real, but not the cause

The publisher loop polls `cudaEventQuery` while the next posted slot's GPU event is pending
(`publisher.cpp:763`, policy in `idle.h`). That was 472,595 calls for 1086 frames in the
decompress process, about 435 per frame, 0.66 s of API time on a spinning thread.

I suspected it slowed the consume thread's own CUDA calls. It does not explain the per-frame
cost: decompress is just as slow alone (finding 6), and `correct` has the same polling with
fast `cudaEventRecord` calls (4 microseconds mean). It still burns a core per stage.

## 8. nsys and UCX tracing

`--trace=ucx` segfaults at `import tango`: the backtrace is in nsys's
`libToolsInjectionMemoryAllocator.so`, called from `libgrpc.so.55` static initialisers during
`dlopen`. UCX is not loaded yet.

| Tried | Result |
|---|---|
| `UCX_MEM_MMAP_HOOK_MODE=none`, `UCX_MEM_MALLOC_HOOKS=n`, `UCX_MEM_CUDA_HOOK_MODE=none`, `UCX_MEM_EVENTS=n` | still crashes, 4 of 4 |
| `nsys profile --attach=<pid>` | option does not exist in 2026.5.1 |
| `--cuda-memory-usage=false` | already the default; still crashes |
| Importing tango later (after CuPy, a GPU call and a sleep) | flaky: 2 of 4 passed, every simpler variant failed |
| `pipeline_device` itself | crashes 3 of 3; libgrpc is a link-time dependency, so there is nothing to delay |
| `--trace=cuda,nvtx`, `--trace=osrt`, `--trace=mpi` | all fine |

Use `--trace=cuda,nvtx` for timelines and `UCX_PROTO_INFO=y` for what UCX is doing.

## 9. One unexplained startup failure

`demo.py ... --scans 10 --network auto --transport-batch 16` once failed with
`RuntimeError: decompress did not become ready`, device logs empty. It ran immediately after a
10-scan scalar run. The same flags passed in two later runs, and the device starts fine by
hand. Cause unknown.

## 10. The benchmark was measuring the demo's verification

`demo.py` checks every volume against a host reference with `np.testing.assert_allclose`, about
0.7 s per 134 MB volume, in the same thread that drains the volume subscription. That is what
made the writer slow and the reconstruct stage wait.

Reference command (512 x 128 x 360, FBP, 10 scans, `--network auto`, scalar):

| | Elapsed | Frames/s | Reconstruct slot wait |
|---|---|---|---|
| verification on (default) | 11.3 s | 320 | 1.0 s |
| `--no-verify-volumes` | 5.9 s | 611 | 0 |

Use `--no-verify-volumes` for any throughput comparison. Every 10-scan number in this file
taken before this finding had verification on.

How far reconstruction may run ahead of the writer is set by the writer's receive ring depth
(`has_room` in `publisher.cpp`), not by the 8 publisher slots.

## 11. Volume writer receives on the GPU

Change in `tango-ucx-gpu-examples/tomography/demo.py` (+38/-2, uncommitted): a small
`DownloadedVolumes` adapter. Outside the tcp profile the writer subscribes with
`memory="cuda:<reconstruct gpu>"`, takes the volume over cuda_ipc, and downloads it with one
pinned copy. Consumers are untouched.

- `UCX_PROTO_INFO=y` before: `cuda/GPU0 to host ... fragmented copy-in copy-out, sysv`.
  After: `cuda/GPU0 to cuda/dev[0] ... zero-copy flushed write to remote, cuda_ipc`.
- The download costs about 12 ms per volume.
- With verification on, elapsed and slot wait are unchanged within noise (host 11.5 to 14.0 s,
  GPU 11.0 to 13.6 s).
- With `--no-verify-volumes`: host 6.77 to 7.01 s (6 runs), GPU 6.39 to 6.56 s (8 of 9 runs).
  About 5%.
- Cost: a 4-volume receive ring on the GPU (512 MiB here) plus one pinned volume per
  reconstructor. A 2-deep ring was tried and rejected: with `--reconstructors 2` or
  `--chains 2` summed slot wait went from 0 to about 10 s.
- Tests: `test_demo.py` 13 OK, `test_output_blocks.py` 9 OK, `test_fan_in.py` 6 OK,
  `test_host_buffering.py` 11 OK, both `gpu_tcp` ctest cases pass. No test exercises the new
  adapter (the ctest demo runs over tcp, the host path).
- Untested: `--network rdma`, a reconstruct GPU other than 0, `--live`.

## 12. Batches larger than 16

Change (uncommitted): one constant, `MAX_BATCH = 512` in `pipeline_control.py`, used by
`demo.py`, the live control validation and `processors.py`; the same bound in
`pipeline_device.cpp` (rebuilt) and `live.html`. `consume_many` now sizes its aligned scratch
for the largest batch seen instead of a fixed 16 rows. 512 is the most that keeps twice a batch
of output slots inside the publisher's 1024.

Decompress stage alone (`probes/decompress_batches.py`, 128 x 512 frames, 3 scans each):

| Batch | Microseconds per frame | Frames/s |
|---|---|---|
| 1 | 1053 to 1090 | 934 |
| 2 | 542 to 578 | 1,773 |
| 4 | 276 to 309 | 3,443 |
| 8 | 145 to 157 | 6,631 |
| 16 | 93 to 100 | 10,286 |
| 32 | 86 to 92 | 11,079 |
| 64 | 89 to 93 | 10,882 |
| 128 | 86 to 93 | 11,164 |
| 256 | 85 to 90 | 11,530 |
| 362 | 85 to 86 | 11,758 |

Scaling is close to linear up to 16 (11x) and flat after it. About 86 microseconds per frame
remains whatever the batch, which is consistent with the per-frame Python work in
`consume_many` (one copy and one nvCOMP array wrapper per frame); not yet profiled.

Whole pipeline, 10 scans, `--network auto --no-verify-volumes`:

| Batch | Elapsed |
|---|---|
| 1 (scalar) | 5.95, 6.56, 6.72 s |
| 16 | 3.55, 3.84, 3.93, 4.08 s |
| 32 | 3.94 s |
| 64 | 3.83 s |
| 128 | 3.63, 3.64 s |
| 256 | 3.38, 3.45, 3.51 s |
| 362 | 3.63 s |

At most about 10% from 16 to 256, and the ranges nearly touch. Decompress alone can now do
about 10,000 frames/s against about 1,000 for the pipeline, so it is no longer the limit. In
these runs the reconstruct stage's FBP took 1.8 to 2.3 s of the 3.4 to 4.1 s, and upstream
stages wait on it.

Batches of 128 and 362 were also run with verification on: volumes verified.

## 13. Target geometry: 2048 x 2048, 2000 projections, 240 frames/s

Measured 2026-10-08 on the RTX 2060 SUPER with the desktop using the GPU (39% before the run).
No repo code changed. Probes: `probes/target_scale.py`, `probes/compress_rates.py`,
`probes/compress_typed.py`; run each with `pixi run python` from `tango-ucx-gpu-examples/`.

Sizes: one uint16 frame is 8 MiB, so 240 frames/s is 1.875 GiB/s (16.1 Gbit/s). A scan of 2000
projections, 50 flats and 50 darks is 2100 frames, 16.4 GiB, 8.75 s. Corrected float32
projections are 31.25 GiB and a 2048-cubed float32 volume is 32 GiB.

| What | Result |
|---|---|
| `fbp_gpu`, 2000 angles x 2048 columns, 4 slices | 68.6 to 69.6 ms per slice: 141 s for 2048 slices |
| `fbp_gpu`, 2000 angles x 1024 columns (2 x 2 binned) | 16.7 to 16.8 ms per slice: 17 s for 1024 slices |
| `fbp_gpu`, 1000 angles x 1024 columns | 9.0 to 9.2 ms per slice: 9.3 s for 1024 slices |
| Dark/flat/-log kernel, 16 frames of 2048 x 2048 uint16 | 3,200 to 6,800 frames/s |
| 2 x 2 sum binning, same frames | 3,300 to 8,200 frames/s |
| Gather 64 rows of every projection from a projection-major uint16 store | 54 to 92 GiB/s |
| Contiguous copy of a 16-frame batch into that store | about 173 GiB/s |

To keep pace with back-to-back scans a GPU must be 16 times this one at full resolution and
2 times at 2 x 2 binning. Correction, binning and copies are not a limit at 240 frames/s.

Where the per-slice time goes (`probes/fbp_split.py`, same day, 60 ms per slice in that run):
filtering is 4.3 to 5.0 ms and the rest, about 92%, is ASTRA backprojection. At 1024 columns
filtering is 1.7 to 2.5 of 15 ms. For comparison, the TomocuPy paper (J. Synchrotron Rad. 2023,
PMC9814072) reports 2048-cubed from 2048 angles, disk read and write included, in 4.4 s
(FourierRec, float32) and 8.1 s (LineRec, direct backprojection) on one A100, and 4.8 s
(FourierRec, float16) on a Quadro RTX 4000. `tomocupy` is not installed in the examples
environment; none of those figures were reproduced here.

nvCOMP 5.3 compression of 16 synthetic frames (a sphere, 20,000 counts, Poisson noise; real
detector data will give other ratios):

| Codec | GiB/s | Ratio |
|---|---|---|
| LZ4, raw bitstream, one chunk per frame (the archive's format today) | 0.20 to 0.22 | 1.00 |
| LZ4, native bitstream | 0.68 to 0.73 | 1.00 |
| LZ4, native, `data_type="<u2"` | 1.30 to 1.40 | 1.00 |
| LZ4, native, bytes shuffled first | 1.21 to 1.25 | 1.23 |
| GDeflate / Zstd, native | 0.52 to 0.57 / 0.44 to 0.45 | 1.19 |
| Cascaded, `data_type="<u2"` | 3.8 to 4.4 | 1.16 |
| Bitcomp, `data_type="<u2"` | 17.5 to 20.9 | 1.53 |

Every LZ4 variant is below the 1.875 GiB/s the detector produces, and plain LZ4 does not
compress these noisy frames at all. Not measured: any of this on an L4 or L40, decompression
rates, real detector frames.

## Corrections to the original rundown

- `astra.projector3d.direct_SIRT`, `create_vol_geom3d`, `create_proj_geom3d` do not exist in
  ASTRA 2.5. `SirtGPU` already uses `direct_FP` / `direct_BP`.
- ASTRA's sinogram layout is (rows, angles, columns), not (angles, rows, columns).
- `UCX_GDR_COPY_FLUSH` and `UCX_MEM_MALLOC_HOOK_MODE` are not UCX 1.22 options. This build has
  no `gdr_copy` or `rc` transport; check `ucx_info -d` on the cluster.
- `cupy.cuda.nvtx.RangePush` is not a context manager; pair it with `RangePop`.
- `cp.cuda.ExternalStream` (`processors.py:31`) is deprecated in CuPy 14 in favour of
  `Stream.from_external()`.

## What I would do next, in order

Revised after findings 10 and 11. Measure everything with `--no-verify-volumes`.

1. **Decide whether to keep the finding 11 change.** It removes the fragmenting for about 5%
   and costs GPU memory.
2. **Use batched decompress at 16 or more** (findings 6, 12). Past 16 it is flat. The limit is
   now the reconstruct stage: for FBP that is finding 2, and for SIRT it is ASTRA's projectors.
3. **On the cluster**, run one scan with `UCX_PROTO_INFO=y` and confirm every GPU-to-GPU link
   says zero-copy. Any `fragmented copy-in copy-out` line from GPU memory is finding 4 again.
4. **Finding 5** (invalid frame with a larger segment) is a latent tango-ucx issue. It does not
   affect cuda_ipc links or default settings; log it for the transport audit.
5. **Polling backoff** (finding 7) is CPU hygiene, not throughput. Low priority.
6. **FBP per-slice overhead** (finding 2) only if FBP is the production algorithm.

## Targeted follow-up

See `TARGETED-FINDINGS.md` for the subsequent metadata-access and FBP-phase
probes, fresh CUDA/NVTX captures, scoped API/copy evidence, and current pipeline
protocol checks. The new evidence separates Python records-view churn from the
C++ compute path and quantifies ASTRA's recurring allocations and memory queries.

The continued investigation is indexed in `FOLLOWUP-SUMMARY.md`, with separate
FBP, metadata, layout, and transport reports. It supersedes the earlier attribution
of oversized SYSV-segment failures to Tango: a raw UCX reproducer establishes a
UCX 1.22 shared-memory length-field overflow. Default segment settings pass;
the tested rendezvous override is not a validated workaround.

`GPU-SATURATION-FOLLOWUP.md` (2026-10-08) explains why the live sliding-window FBP run stays
near 57% GPU under MPS: one reconstruct thread, host-bound in the per-slice loop of finding 2.
