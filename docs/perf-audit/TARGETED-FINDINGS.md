# Targeted tomography bottleneck follow-up

Measured 2026-10-07 on the local RTX 2060 SUPER, CuPy 14.2, ASTRA 2.5.0.
Two isolated probes and one three-scan CUDA/NVTX pipeline capture were added.
Application source and existing working-tree edits were preserved.

## Priority: ASTRA's per-slice FBP work

`probes/fbp_phases.py` calls the production `fbp_gpu`, varying slice count and
filter batch size. Uploads and independent ASTRA reference reconstruction occur
outside the timed calls. Every case passes `rtol=3e-4, atol=2e-6`, and the input is
checked unchanged. Maximum absolute reference error was below `3.8e-8`.

Unprofiled medians, three repeats after warmup, 360 angles, 512 detector columns,
Parzen filter:

| Slices | Filter batch 1 | Filter batch 32 |
| --- | ---: | ---: |
| 1 | 2.42 ms | 2.08 ms |
| 8 | 15.48 ms | 12.72 ms |
| 32 | 49.64 ms | 44.89 ms |
| 128 | 200.06 ms | 174.81 ms |

Filtering batches help, but do not batch backprojection. For 128 slices and a
filter batch of 32, the instrumented call makes four FFT pairs, two GPU data
links, one algorithm creation, and **128 `astra.algorithm.run` calls**. Increasing
the input block size cannot amortize this per-slice work. The application's
normal filter limit is `1 << 24` padded samples, about 45 slices at this shape;
the table deliberately fixes the filter batch for controlled comparison.

Nsight evidence restricted to `astra.run` ranges inside
`instrumented.rows128.filter32` in `fbp-phases.nsys-rep`:

| API | Calls | Summed host API duration |
| --- | ---: | ---: |
| `cudaMallocPitch` | 256 | 14.36 ms |
| `cudaFree` | 256 | 30.01 ms |
| `cudaMemGetInfo` | 1024 | 18.97 ms |
| `cudaStreamSynchronize` | 256 | 56.84 ms |
| `cudaStreamCreate` / `cudaStreamDestroy` | 128 each | 0.89 / 0.66 ms |

The same calls launch 2944 `devBP` kernels, totaling 58.80 ms of GPU activity.
The instrumented `astra.run` wall duration is 149.40 ms. Synchronization waits
include kernel execution; these figures must not be added as independent costs.
They establish substantial recurring API work even though the Python algorithm
and links are already reused across slices.

GPU linking does not eliminate ASTRA's internal copies: 256 device-to-device
copies move 228,589,568 bytes in 1.05 ms of GPU copy activity. Another 512 small
host-to-device copies move 737,280 bytes of parameters. There are no
device-to-host payload copies in these scoped runs. The large cost is the
recurring orchestration, rather than the device copy duration alone.

Next experiment: preserve ASTRA's 2D backprojection math while reusing its native
pitched buffers, stream, texture resources and geometry across slices, or add a
native slice-batch entry point. Merely caching Python links/algorithm objects
does not remove the measured internal allocations. Any alternate 3D projector
needs fresh scientific validation; prior audit notes report interpolation
differences, and this follow-up does not validate a replacement. ASTRA's
[2D FBP implementation](https://raw.githubusercontent.com/astra-toolbox/astra-toolbox/v2.5.0/cuda/2d/fbp.cu)
also separates optional filtering from its backprojection call.

## Python metadata churn is real, but scope matters

`probes/metadata_access.py` retains real host UCX batches of 1, 16 and 128 frames
from the existing opaque publisher. The records views share the same address,
share memory, and are read-only: the payload/record storage is not copied by
repeated property access. This probe uses the fixture's seed metadata schema,
not the tomography device's 40-byte application fields.

| Access pattern | Batch 16, µs/frame | Batch 128, µs/frame |
| --- | ---: | ---: |
| `batch.records` again for every frame | 40.51 | 40.47 |
| `batch.records` once, then iterate | 3.39 | 0.91 |
| Iterate previously cached records | 0.66 | 0.57 |

Constructing the property alone costs 35–37 µs/access, largely independent of
batch length. In the current source, `python/_core.cpp:254` constructs a NumPy
structured dtype, and `PythonBatch::records` invokes it on every access.
Caching the dtype per immutable description, or using one records view per batch
in consumers, is a focused Python optimization candidate.

The tomography compute stages use the C++ `Batch::records()` pointer directly
(`pipeline_device.cpp:333`), so this result does **not** establish that Python
dtype churn is their bottleneck. The wire path encodes a fixed binary header into
preallocated operation storage (`publisher.cpp:509`, `wire.cpp:345`); Tango command
discovery and JSON negotiation happen at subscription setup. Receiver batching
does not coalesce sends: the publisher still calls `ucp_am_send_nbx` per frame.
Native per-frame messaging/credit overhead remains a separate hypothesis, not a
measured 40 µs/frame cost.

The probe uses the staged extension recorded in `metadata-access.json`. It lacks
the current source's `Batch.index` Python API, so staged binaries and current
source are not identical. The optional native-index comparison is skipped when
that method is unavailable. Connection times were 1.23 s initially, then 26–28 ms;
the first includes lazy imports and must not be interpreted as discovery alone.

## Current pipeline: GPU protocols are selected correctly locally

The three-scan capture uses 128 × 512 detector frames, 360 angles, FBP Ram–Lak,
transport/processing batch 16, automatic networking and `--no-verify-volumes`.
All 1086 input frames were processed and archived, and all three volumes completed
with no stage failure or quarantined bytes. Acquisition elapsed 2.373 s under
profiling; this short run is not a steady-state throughput benchmark.

| Named NVTX operation | Calls | Summed host duration |
| --- | ---: | ---: |
| Decompression `consume_many` | 71 | 191.96 ms |
| Correction `consume` | 1086 | 217.34 ms |
| Reconstruction `consume` | 1080 | 1218.85 ms |
| Reconstruction `reconstruct` | 3 | 1007.63 ms |

`reconstruct` is nested inside the final `consume` of each scan; do not sum those
two rows. Correction waited 715 ms for output slots while reconstruction's output
slot wait was only 6 µs. This is consistent with reconstruction exerting upstream
backpressure. Stage durations run in different processes and must not be summed
into a pipeline elapsed time or a utilization percentage.

`UCX_PROTO_INFO=y` confirms CUDA-to-CUDA rendezvous uses `cuda_ipc/cuda` with
zero-copy flushed write for decompression output, correction output and volume
delivery. Endpoint health lists both `sysv/memory` and `cuda_ipc/cuda`; that list
alone does not identify which protocol carries each memory type.

Source host memory to the decompression GPU still uses fragmented copy-in/out:
9314 host-to-device copies, median 8240 bytes, 71.59 MB total, 7.29 ms of GPU copy
activity. Decompression also does an intentional copy into aligned nvCOMP scratch.
Correction-to-sinogram assembly intentionally writes the frame into the
`(rows, angles, columns)` layout; the receive frame is `(rows, columns)` and its
sinogram destination is strided. These are ownership/layout boundaries to measure,
not evidence of an accidental DLPack host fallback.

The volume writer receives on GPU and then copies for host output. Trace-wide
host-process copies include preparation, reference creation and final output;
they cannot all be attributed to timed acquisition. GPU protocol selection on
this one machine does not establish RDMA or GPUDirect selection on the cluster.

## Reproduction and evidence

Run from `tango-ucx-gpu-examples/` with `pixi run` so CUDA headers are discoverable:

```bash
pixi run python docs/perf-audit/probes/metadata_access.py --output /tmp/metadata-access.json
pixi run python docs/perf-audit/probes/fbp_phases.py --output /tmp/fbp-phases.json

/opt/nvidia/nsight-systems/2026.5.1/target-linux-x64/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none -o /tmp/fbp-phases \
  pixi run python docs/perf-audit/probes/fbp_phases.py \
  --rows 32 128 --filter-rows 32 --repeats 1 --output /tmp/fbp-profiled.json
```

Baseline medians: `fbp-phases.json`. Metadata: `metadata-access.json`. Scoped
SQL, counts and timing evidence: `targeted-trace-evidence.json`. Native reports:
`traces/fbp-phases.nsys-rep` and `traces/pipeline-targeted.nsys-rep`. Pipeline
summary and per-device protocol logs: `pipeline-targeted/`.

Report analysis uses the installed Nsight skill's `report-describe`,
`report-query`, `report-fact` and `report-doctor`, with its bundled interpreter.
Queries in the evidence file can be replayed through `report-query`. Kernel/API
correlation health checks pass. The pipeline has NVTX on processing threads only;
transport threads are unlabelled, and startup/preparation gaps are present.
Neither capture includes CPU sampling or GPU hardware metrics. No CPU saturation,
SM utilization or kernel-internal stall claim is made. `--trace=ucx` was avoided
because the earlier audit reproduced a Tango/trace-injection crash.

The highest-value next probe is persistent native 2D FBP resources with the same
numerical checks. After that, benchmark native UCX frame/header processing
separately from Python wrappers, and capture actual cluster protocol choices.
