# FBP resource reuse: 1.17× local pipeline throughput

Measured 2026-10-07 on the local RTX 2060 SUPER, ASTRA 2.5.0 and CuPy 14.2.
Application source files are unchanged; candidate pipeline measurements use an
experiment-only import shim.

## Full pipeline: 1.17× throughput, 14.9% less acquisition time

`probes/fbp_pipeline.py` compares the unchanged demo with
`probes/fbp_shim/sitecustomize.py`, an import shim that replaces only
`processors.fbp_gpu` in the embedded reconstruction
process. It retains native geometry for that process, serializes native calls,
restricts the experiment to GPU 0 and `(128, 360, 512)` sinograms, and logs every
completed native call. Production files are unchanged.

The unprofiled comparison uses 10 scans per run, 128 slices, 512 pixels,
360 projections, Ram–Lak, automatic networking, GPU sinogram storage, whole-volume
output and transport/processing batch 16. Per-volume reference verification is
disabled in both methods. Both use the production filtering cap of `1 << 24`
padded samples, which gives **45 slices per filter block** at this shape.
Baseline/candidate order alternates between the two pairs.

| Pair | Production acquisition | Persistent acquisition | Throughput ratio |
| --- | ---: | ---: | ---: |
| 0, production first | 3.1095 s | 2.6787 s | 1.161× |
| 1, persistent first | 3.2288 s | 2.7158 s | 1.189× |
| Mean elapsed | **3.1692 s** | **2.6972 s** | **1.175×** |

The paired mean acquisition time falls **14.9%**, equivalent to **17.5% higher
throughput** at this fixed scan count. Observed production throughput is
3.10–3.22 volumes/s; persistent native throughput is 3.68–3.73 volumes/s.
The reconstruction-stage accumulated counter falls from a mean 1.6608 s to
0.9179 s, or 44.7%. Those stage times overlap upstream/downstream activity and
must not be added to acquisition elapsed. The smaller acquisition gain shows
why the isolated backend speedup cannot be assumed to multiply pipeline throughput.

Every timed run completes 10 volumes and archives 3620 frames, with no stage
failure, input failure or quarantined bytes. Both candidate logs confirm 10
completed native calls and 45-row filtering. Reference creation remains in the
demo's untimed preparation; transport, volume download, volume output and its
normal acquisition workflow remain active. This is a short-run local comparison
on a shared GPU, with uncontrolled clocks and two pairs, not a cluster result
or a long-run steady-state throughput estimate.

A separate two-scan candidate run with **normal volume verification enabled**
passes the demo's independent ASTRA reference comparison for both delivered
volumes. It archives 724 frames and reports relative phantom L2 error 0.124569.
That run includes first-use native compilation and verification work and is
excluded from the throughput table.

Evidence: `fbp-pipeline-paired/results.json` and each run's `summary.json`,
`reconstruct.log` and device logs; verification evidence is in
`fbp-pipeline-verified/`. The parent shell had no inherited `UCX_*`, `TANGO_UCX_*`,
`CUDA_VISIBLE_DEVICES`, `PYTHONHOME` or `PYTHONPATH` overrides. The runner clears
inherited profiling-shim `PYTHONPATH` and `UCX_PROTO_INFO`, then adds the candidate
shim only for candidate runs. Automatic networking sets `UCX_TLS=all` for both.

## The strongest result is direct native 2D backprojection

The native library already exports a raw-pointer `astraCUDA::BP` entry point.
It can backproject the padded CuPy FFT result directly into the caller's output
slice. This removes the linked scratch image, linked scratch sinogram, per-slice
pitched allocations, staging copies and memory-availability queries measured in
the previous audit, while retaining ASTRA's original 2D kernels and texture
interpolation.

`probes/fbp_native.cu` supplies a small C interface to that function and retains
ASTRA-generated geometry between calls. Its `persistent_2d` variant runs the
publicly exported raw-pointer BP once at the beginning of each filter batch to
initialize ASTRA's angle constants. It then calls the private exported
`BP_internal` for subsequent slices, reusing the caller's stream and those
constants. This still creates and destroys a texture and synchronizes once per
slice. It is a resource-reuse experiment, not a complete native batch API.

Seven unprofiled repeats after warmup; 360 angles, 512 columns, Parzen, filtering
32 slices at a time. Upload, compilation, persistent geometry initialization and
independent reference calculation are outside timing. Filtering, backprojection
and completion synchronization are inside timing.

| Slices | Production FBP | Direct native BP | Persistent native BP | Custom slice-batch BP |
| --- | ---: | ---: | ---: | ---: |
| 32 | 34.59 ms | 17.93 ms | **17.35 ms** | 19.98 ms |
| 128 | 118.20 ms | 71.75 ms | **68.66 ms** | 79.77 ms |

At 128 slices the persistent path reduces the paired median by **41.9%**, or
**1.72×**. Of the 49.54 ms difference, removing the high-level data/algorithm
wrapper accounts for most of the gain: direct native BP already saves 46.44 ms.
Further stream/constant reuse saves another 3.10 ms in this run.

The previous audit's production baseline was 174.81 ms. Later runs in this
session ranged from about 118 to 154 ms, so use the table's same-run comparison,
not a ratio against the older capture. GPU clocks were not locked; methods run
in a fixed order, and this is an isolated backend benchmark rather than measured
application throughput.

## Nsight confirms the removed orchestration

`traces/fbp-native.nsys-rep` contains all four implementations. The following
counts use only each implementation's `rows128.parzen.repeat1` NVTX range.
They include the filtering calls and final synchronization. This is a different
scope from the earlier report, which restricted API counts to nested ASTRA runs.

| Timed operation | Production | Direct native | Persistent native | Custom batch |
| --- | ---: | ---: | ---: | ---: |
| `cudaMallocPitch` | 256 | 0 | 0 | 0 |
| `cudaFree` | 256 | 0 | 0 | 0 |
| `cudaMemGetInfo` | 1024 | 0 | 0 | 0 |
| `cudaMemcpy2D` | 256 | 0 | 0 | 0 |
| `cudaStreamCreate` | 129 | 128 | 4 | 0 |
| `cudaMemcpyToSymbolAsync` | 512 | 512 | 16 | 4 |
| `cudaCreateTextureObject` | 128 | 128 | 128 | 4 |
| `cudaStreamSynchronize` | 386 | 261 | 137 | 5 |
| BP kernel launches | 2944 | 2944 | 2944 | 92 |
| Summed BP GPU activity | 63.23 ms | 59.97 ms | 59.94 ms | 73.07 ms |

Production has 384 device-to-device payload copies totaling 362,807,296 bytes:
256 internal staging copies plus 128 scratch-image-to-output copies. The direct
and persistent paths have **no device-to-device copies** in their timed scopes.
The custom batch has four small device-to-device constant uploads totaling
23,040 bytes. All methods retain a 2,052-byte host upload for the filter response;
the persistent method also uploads angle constants at each filter-batch start.

Production API durations include 18.53 ms in memory queries, 11.95 ms in pitched
allocations and 8.53 ms in frees. Synchronization includes GPU execution and must
not be added to GPU activity as an independent cost. Scoped trace wall times are
134.26 / 75.39 / 70.32 / 79.87 ms respectively; use unprofiled medians for speedup.

The installed Nsight skill's `report-doctor` confirms 100% kernel/runtime
correlation and consistent GPU attribution. There are no GPU hardware metrics
or CPU samples; this report does not diagnose a kernel-internal stall cause.
Full bounded queries and gateway results are in `fbp-native-trace-evidence.json`.

## Fewer launches alone do not establish a faster backend

The custom `batched_2d` prototype uses one pitched 2D texture spanning the filter
batch, one CUDA grid z dimension per slice and 16-angle accumulation groups.
It preserves the original centered coordinates, ASTRA-generated parallel
geometry coefficients, border addressing, linear texture interpolation and
FBP scaling. A 32-slice filter block therefore needs 23 BP launches rather than
736. Four such blocks reduce the 128-slice launch count from 2944 to 92.

It passes the reference checks but its GPU BP work takes 73.07 ms, compared with
59.94 ms for persistent native ASTRA. Its unprofiled overall time is also slower:
79.77 versus 68.66 ms. Keep the native ASTRA resource-reuse path as the immediate
candidate. A true batch kernel needs separate optimization evidence before it
would justify replacing ASTRA's existing kernels. The custom kernel's earlier
global-coefficient variant was slower still; current measurements use constant
memory. No Nsight Compute metrics were gathered.

## Numerical coverage and practical limits

The current probe was checked in 48 method/shape/filter cases across these runs:

- `fbp-native-persistent.json`: 32 and 128 slices, 360 angles, 512 columns,
  Parzen; all four implementations.
- `fbp-native-odd-signed.json`: 1 and 8 slices, 31 irregular angles, 65 columns,
  signed inputs; Ram–Lak, Shepp–Logan, Hann and Parzen; all four implementations.
- `fbp-native-cutoff-edge.json`: 8 slices, 181 irregular angles, 511 columns,
  positive/negative detector-edge impulses; Shepp–Logan and Hann with cutoff
  0.7; all four implementations.

Every case passes independent host ASTRA `FBP_CUDA` reference checks at
`rtol=3e-4, atol=2e-6`, and input bytes remain unchanged. Maximum absolute error
over these runs is `3.82e-8`. The custom and native methods exhibit the same
reported maximum reference error as production in the tested cases. This covers
orientation, odd dimensions, detector borders and filter/cutoff behavior; it
does not prove equivalence for arbitrary geometries or supersampling.

The prototype deliberately remains outside production:

- `BP_internal` is a private exported C++ symbol, absent from the installed
  public header. Its ABI and the raw-pointer interfaces are tied to the measured
  ASTRA 2.5.0 installation. Production should obtain a supported stream/batch API
  or a version-pinned native integration with compatibility checks.
- ASTRA's angle constants are shared within the CUDA context. Calls with differing
  geometries on concurrent threads/streams must be serialized; the prototype
  assumes one reconstruction at a time. The custom kernel has its own shared
  constant arrays and the same concurrency limitation.
- The native build targets `sm_75` only. This is the local GPU's architecture,
  not a portable cluster binary.
- The harness accepts contiguous float32 GPU data, unit detector/voxel spacing,
  square output slices, parallel geometry and no supersampling. It supports at
  most 2560 angles. Pitched texture width, pitch, alignment and height must satisfy
  the device's limits; the custom texture height is `filter_rows × angles`.
- The native context retains geometry and coefficient storage only. Borrowed
  input/output pointers exist only inside `fbp_run`, which completes all work
  before return; no transport slot is retained. Filter scratch remains bounded
  by the selected filter block size.

## Reproduction

From `tango-ucx-gpu-examples/`:

```bash
pixi run python docs/perf-audit/probes/fbp_native.py \
  --rows 32 128 --repeats 7 --output /tmp/fbp-native.json

pixi run python docs/perf-audit/probes/fbp_pipeline.py \
  --scans 10 --repeats 2 --output /tmp/fbp-pipeline-paired

pixi run python docs/perf-audit/probes/fbp_pipeline.py \
  --scans 2 --repeats 1 --candidate-only --verify \
  --output /tmp/fbp-pipeline-verified

pixi run python docs/perf-audit/probes/fbp_native.py \
  --rows 1 8 --angles 31 --columns 65 \
  --filters ram-lak shepp-logan hann parzen --pattern signed \
  --irregular-angles --repeats 1 --output /tmp/fbp-native-odd.json

pixi run python docs/perf-audit/probes/fbp_native.py \
  --rows 8 --angles 181 --columns 511 --filters shepp-logan hann \
  --cutoff 0.7 --pattern edge-impulse --irregular-angles --repeats 1 \
  --output /tmp/fbp-native-edge.json

/opt/nvidia/nsight-systems/2026.5.1/target-linux-x64/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none -o /tmp/fbp-native \
  pixi run python docs/perf-audit/probes/fbp_native.py \
  --rows 128 --repeats 1 --output /tmp/fbp-native-profiled.json

/opt/nvidia/nsight-systems/2026.5.1/target-linux-x64/python/bin/python \
  docs/perf-audit/probes/fbp_trace.py --report /tmp/fbp-native.nsys-rep \
  --output /tmp/fbp-native-trace-evidence.json
```

The native probe compiles its adjacent `.cu` to `_fbp_native.so` automatically with the active
pixi environment's CUDA compiler and ASTRA headers/library. Upstream implementation
references: [FBP scaling and dispatch](https://github.com/astra-toolbox/astra-toolbox/blob/v2.5.0/cuda/2d/fbp.cu),
[native BP constants, texture and stream lifecycle](https://github.com/astra-toolbox/astra-toolbox/blob/v2.5.0/cuda/2d/par_bp.cu).
