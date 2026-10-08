# Grouped scan assembly with asynchronous double buffering

Measured on 2026-10-07 on the local RTX 2060 SUPER, with ASTRA 2.5.0,
CuPy 14.2 and an AMD Ryzen 5 2600 (six cores, twelve logical CPUs).

The experiment demonstrates genuine next-scan assembly and some GPU kernel
overlap with reconstruction. It does **not** establish a repeatable improvement
in verified throughput. Current paired capacity measurements improve about
3.6–3.8%; verified mean improvements are 0.6% and 3.0%, below the observed
between-run variation. Keep the combined path experimental.

Production source files were not changed. The prototype uses a generated copy
of the current C++ device, a separate reconstruction worker, two owned sinograms
and distinct nonblocking assembly/reconstruction streams. Reconstruction remains
serial. This is a scheduling experiment, with no geometry, filtering, tolerance,
iteration, transport-setting or output-policy changes.

## Compare with the previous runs first

The requested initial comparison retains the earlier FBP pipeline settings:
ten scans, 128 rows, 360 projections, 512 columns, Ram–Lak, GPU sinograms,
whole-volume output, one chain/worker, all roles on GPU 0, automatic networking,
and transport/processing batches of 16. Independent per-volume verification is
disabled for these capacity comparisons, as in the previous runs. Reference
preparation, volume downloading/writing and the required compressed archive
remain active. Persistent native FBP retains the production filtering cap and
45-row filter blocks.

New combined-path runs repeat three times, alternating backend order:

| FBP backend | Previous scalar mean acquisition | New grouped/two-buffer mean acquisition | Previous volumes/s | New volumes/s |
| --- | ---: | ---: | ---: | ---: |
| Production | 3.1692 s | 3.9550 s | 3.155 | 2.528 |
| Persistent native | 2.6972 s | 3.4523 s | 3.708 | 2.897 |

This historical comparison looks about 20–22% slower, but a fresh scalar baseline
also slowed substantially. Historical numbers alone cannot attribute a
regression to the new path. The cause of this baseline drift was not isolated;
GPU clocks, CPU frequency and desktop activity were uncontrolled. There were no
other GPU compute benchmarks running concurrently.

Evidence: [previous runs](fbp-pipeline-paired/results.json),
[new candidate runs](scan-overlap-historical/results.json).

## Contemporaneous paired capacity measurements

Two additional pairs per backend use the same generated binary for scalar and
combined paths. Order alternates scalar/candidate and candidate/scalar. Rates
are total completed scans divided by total acquisition time, equivalent to ten
scans divided by mean elapsed time; they are not arithmetic means of run rates.

| FBP backend | Scalar mean acquisition | Combined mean acquisition | Scalar volumes/s | Combined volumes/s | Throughput change | Individual pair changes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Production | 4.0526 s | 3.9055 s | 2.468 | 2.560 | +3.77% | +3.53%, +4.00% |
| Persistent native | 3.2974 s | 3.1825 s | 3.033 | 3.142 | +3.61% | +10.95%, −3.94% |

The production capacity result is small and positive in both pairs. Persistent
native capacity is inconsistent across pairs. Neither capacity table measures
verified acceptance. Grouping and overlap are combined here; no grouped-only
control was measured, so their contributions cannot be separated.

Reconstruction's accumulated wall counter rises from 2.3421 to 2.5188 s for
production and from 1.1886 to 1.4276 s for persistent native in these pairs.
Concurrent assembly therefore has a cost inside the reconstruction envelope,
even though application elapsed time falls slightly. These overlapping counters
cannot be added to acquisition time or used to establish GPU/CPU saturation.

Evidence: [paired capacity runs](scan-overlap-paired/results.json).

## Verified throughput

Two further alternating pairs per backend enable the demo's normal independent
ASTRA reference comparison and phantom-error checks on **every** volume. The
required archive and volume writer are retained. Each run completes ten volumes
and archives 3,620 frames without stage/input failures or quarantined memory.
All forty combined-path volumes and forty scalar volumes pass verification;
relative phantom L2 error remains 0.124569.

| FBP backend | Scalar mean acquisition | Combined mean acquisition | Scalar verified volumes/s | Combined verified volumes/s | Mean change | Individual pair changes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Production | 11.9271 s | 11.8554 s | 0.8384 | 0.8435 | +0.60% | +0.38%, +0.82% |
| Persistent native | 11.2380 s | 10.9138 s | 0.8898 | 0.9163 | +2.97% | −2.25%, +8.01% |

Scalar verified rates range from 0.8160 to 0.8621 volumes/s for production and
0.8332 to 0.9548 for persistent native. Mean changes are smaller than this
variation; two short pairs do not establish a reliable verified-throughput win.
The much lower verified rates also show that the combined scheduling change
does not resolve the existing output/verification bottleneck.

These are the existing demo's fill/drain-inclusive acquisition intervals.
Per-volume verification and volume writing are inside the interval, and the
archive process drains before its end. Reference preparation, final archive
byte comparison, plots and summary/cleanup are outside it. No new warmed
steady-state interval or changed verification policy is implied.

Evidence: [verified paired runs](scan-overlap-verified-paired/results.json),
[initial three-volume checks](scan-overlap-verified/results.json).

## What the trace establishes

A separate CUDA/NVTX capture runs four scans with each backend. Its profiled
elapsed times are excluded from every throughput table. Assembly launches
23 layout-copy kernels per scan rather than 360 per-frame copies.

| FBP backend | Assembly GPU activity, four scans | Next-scan assembly groups overlapping preceding BP | Summed direct assembly/BP interval overlap |
| --- | ---: | ---: | ---: |
| Production | 2.418 ms | 6 of 69 eligible groups | 0.0175 ms |
| Persistent native | 2.507 ms | 25 of 69 eligible groups | 0.0963 ms |

The first scan has no preceding reconstruction; its 23 groups are excluded from
the eligible-group denominator. Overlap queries require consecutive scan IDs,
the same process/device context, different CUDA streams, and intersecting
assembly-copy and ASTRA `devBP` kernel intervals. All 12,452 production-process
kernels and 11,940 persistent-process kernels have matching runtime correlation.
This is device activity evidence, beyond a host flag saying reconstruction is
active. Only a small amount of assembly GPU work directly overlaps BP in this
capture; receiving and host submission can also progress during reconstruction.

Production still performs its existing per-slice allocations and handoffs.
Persistent native still synchronizes its reconstruction stream per slice, but
uses a separate stream from assembly. Synchronization durations include GPU
work and are not additional independent overhead. There are no CPU samples,
GIL events or GPU hardware counters establishing a contention/saturation cause.

Evidence: [trace queries and results](scan-overlap-trace-evidence.json),
[Nsight report](traces/scan-overlap.nsys-rep).

## Ownership, memory and correctness

- C++ receives on the assembly stream, splits groups at scan boundaries and
  noncontiguous payload addresses, and validates frame order, identity,
  calibration and byte length. Python validates projection continuity and
  angles before enqueueing one grouped copy.
- The receive view remains alive through its last queued assembly read. Its
  existing transport completion event releases credits on that stream; borrowed
  receive pointers are never retained as future jobs.
- A per-buffer ready event orders reconstruction after assembly. A buffer
  becomes reusable only after reconstruction/postprocessing completes. Output
  slots are acquired and published by the reconstruction worker with its own
  stream completion, and queued work drains before buffers are destroyed.
- Two sinograms retain 180 MiB at the audited shape, **90 MiB more** than scalar.
  All combined runs report at most two busy buffers and one ready scan. Existing
  transport/output ring sizes are retained. Whole-device memory peaks were not
  sampled; the pool counter accounts only for owned sinograms.
- Metadata, partial groups, scan transitions and ring reuse are exercised by
  the full-pipeline runs. The explicit noncontiguous split counter stays zero
  because those delivered groups are contiguous. That branch therefore lacks
  a direct pipeline trigger in these runs.
- Three focused GPU tests reject malformed/oversized/incomplete groups without
  ingestion mutation, compare distinct signed-input scans and filter/smoothing
  settings against an independent reference, preserve input bytes, and verify
  buffer reuse with reconstruction on another host thread. They use four rows,
  31 angles and 65 columns.
- A further independently verified full-pipeline check uses eight rows,
  97 angles and 64 columns, batch 16, and three scans. All three volumes and
  297 archived frames pass; neither 97 projections nor 99 source frames divides
  by 16. This is a correctness check, not a performance comparison.

The prototype supports GPU-retained sinograms, whole-volume FBP output, one
chain and one reconstruction worker. It is not a production implementation for
SIRT, host buffering, block output, multiple GPUs/workers or concurrent changing
geometries. The persistent backend retains the existing shim's fixed geometry,
GPU-0 restriction, private ASTRA ABI and local `sm_75` build. No local result
establishes a target-cluster speedup or justifies tuning batch sizes for this GPU.

Evidence: [GPU tests](probes/test_scan_overlap.py),
[second-geometry pipeline check](scan-overlap-odd-verified/results.json).

## Reproduce

From `tango-ucx-gpu-examples/`, build the isolated device and run the candidate:

```bash
pixi run python docs/perf-audit/probes/scan_overlap_pipeline.py \
  --build-only --output /tmp/scan-overlap-build

pixi run python docs/perf-audit/probes/test_scan_overlap.py

pixi run python docs/perf-audit/probes/scan_overlap_pipeline.py \
  --device-server /tmp/scan-overlap-build/build/pipeline_device \
  --scans 10 --repeats 3 --output /tmp/scan-overlap-candidate

pixi run python docs/perf-audit/probes/scan_overlap_pipeline.py \
  --device-server /tmp/scan-overlap-build/build/pipeline_device \
  --scans 10 --repeats 2 --with-baseline --verify \
  --output /tmp/scan-overlap-verified
```

The runner defaults to production and persistent native FBP. Select
`--backends production` for other geometries. `--buffers 1` is available for a
future grouped-only control, but was not measured here. The build retains the
generated C++ source, input hashes, CMake configuration and compiler log;
results retain commands, stage reports and GPU state before each run.

Mechanism capture and bounded analysis:

```bash
/opt/nvidia/nsight-systems/2026.5.1/target-linux-x64/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o /tmp/scan-overlap \
  pixi run python docs/perf-audit/probes/scan_overlap_pipeline.py \
    --device-server /tmp/scan-overlap-build/build/pipeline_device \
    --scans 4 --repeats 1 --output /tmp/scan-overlap-profiled

/opt/nvidia/nsight-systems/2026.5.1/target-linux-x64/nsys export \
  --type sqlite --output /tmp/scan-overlap.sqlite /tmp/scan-overlap.nsys-rep

pixi run python docs/perf-audit/probes/scan_overlap_trace.py \
  --database /tmp/scan-overlap.sqlite --output /tmp/scan-overlap-evidence.json
```

The actual capture additionally requested `python-gil`; it produced no GIL
event table, so no GIL diagnosis is made. All throughput measurements are
unprofiled. No production defaults, branches or commits were changed.
