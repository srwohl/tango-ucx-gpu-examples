# Pipeline throughput: take-home priorities

## What the work supports

The strongest measured incremental FBP candidate is persistent native resource
reuse: about 17.5% more volumes/s in the paired local pipeline experiment.
The strongest architectural opportunities to investigate are a faster verified
output path, reconstruction on independent GPUs, and overlapping scan assembly
with reconstruction. Placement and replication have not yet been demonstrated
to improve this pipeline on the target cluster.

All existing measurements are local to one RTX 2060 SUPER, with the stages
sharing that GPU. The FBP throughput comparisons disabled per-volume reference
verification; separate candidate runs passed normal independent verification.
Neither those rates nor isolated speedups establish sustained verified
throughput on the cluster.

Use batched decompression as the comparison baseline. Processing and transport
batches of 16 already moved its isolated capacity to roughly 10,000 frames/s,
versus roughly 1,000 frames/s for the local pipeline. Larger batches offer little
additional decoder improvement and only uncertain, modest application gains.

## Ranked work to investigate

| Priority | Concrete next action | Evidence and expected mechanism | What would count as success |
| --- | --- | --- | --- |
| 1 | Make downloading, numerical verification and required writing a bounded output service. Compare a pooled CPU verifier with GPU comparison/reductions using the independent reference. | Inline host verification previously took about 0.7 s per 128 MiB volume and roughly halved observed throughput. The current reader also downloads synchronously and writes before its next read. Decoupling releases transport credits sooner; faster checks or sufficient verifier capacity are needed for sustained gains. | Higher **verified and accepted volumes/s** after all mandatory validation/output drains, with stable queue occupancy, unchanged tolerances and failure detection. |
| 2 | Harden persistent native FBP as an opt-in backend, retaining production filtering. | Paired acquisition mean 3.1692 to 2.6972 s, about 17.5% more throughput; isolated reconstruction improved 1.72 times. Reusing buffers and geometry removes recurring allocations and staging. | Paired sustained verified-throughput improvement on target GPUs and representative geometries; independent numerical checks still pass. |
| 3 | Put reconstruction on independent GPUs; distribute complete scans to a reconstruction pool. Compare this with a complete local processing chain on each GPU. | Reconstruction exerts upstream backpressure after decoder batching. Independent GPU capacity is a plausible larger source of aggregate volumes/s; local complete chains send compressed scans instead of expanded corrected projections across the interconnect. No scaling factor is established. | Additional GPUs raise aggregate verified volumes/s and useful output MB/s; per-GPU efficiency, source/archive load, latency and queue depth remain acceptable. |
| 4 | Evaluate grouped ingestion and the experimental two-buffer assembly path on the target deployment before production integration. | Isolated grouped ingestion measured 37.40 to 2.77 ms/scan at batch 16. A generated-device prototype now demonstrates genuine next-scan GPU assembly/BP overlap; local paired capacity rises about 3.6–3.8%, while verified mean changes of 0.6%/3.0% remain within run variation. It adds 90 MiB of owned sinogram memory. | A repeatable verified-throughput gain beyond persistent FBP and run variability, bounded memory, and correct scan/calibration/ring-wrap handling. |
| 5 | Reduce expanded-data boundaries where they become material: colocate or fuse decompression/correction, and consider correction directly into grouped sinogram assembly. | Each separate stage uses output and receive rings, GPU movement and per-frame dispatch. Local GPU transport is already CUDA IPC; fusion could remove a boundary, but extra coupling may reduce useful overlap. Host sinogram retention adds download/reupload traffic. | Fewer measured bytes/copies and lower memory/dispatch cost accompany a rise in accepted volumes/s; capacity and failure semantics remain valid. |
| 6 | Validate cluster transfer paths, GPU/HCA/NUMA placement and CPU allocation before transport tuning. | Local GPU protocols are selected correctly. Actual RDMA, GPU memory registration and cross-GPU paths remain unmeasured. Polling uses CPU cores, but was refuted as the cause of scalar decoder latency. | The intended protocol is actually selected, transfers meet demand with CPU/NIC headroom, and any tuning raises sustained application throughput. |

For SIRT, replace priority 2 with reconstruction capacity and scientifically
validated projector/algorithm work: the measured SIRT path spends about 92% of
wall time in ASTRA kernels. Small handoff optimizations are unlikely to help it.
Reducing iterations, angles or resolution changes the scientific workload and
must be evaluated as a quality/throughput tradeoff, not an equal-work speedup.

## Architecture experiments worth running first

Compare these placements with the same input, reconstruction settings,
verification and required sinks:

1. All processing roles on one GPU: reference placement.
2. Decompression/correction together on GPU 0, reconstruction on GPU 1:
   separate preprocessing work from reconstruction, measuring the added
   corrected-projection transfer cost.
3. Shared preprocessing plus a complete-scan reconstruction pool on distinct
   GPUs: exploit additional reconstruction capacity before replicating a
   decoder that already has headroom.
4. One complete chain per GPU, each assigned different compressed scans:
   compare avoiding expanded inter-GPU/network traffic against preprocessing
   contention within each reconstruction GPU.

The existing demo does **not** assign different GPUs to indexed workers.
`device_command` gives every worker the same configured role GPU, so
`--reconstructors 2` or `--chains 2` alone is not a two-GPU test. A per-worker
launch arrangement and corresponding output receive placement are required.
Existing pull sets also require complete-scan receive rings and divisibility by
the transport batch: 360 projections and batch 16 fail the reconstructor check;
362 input frames and batch 16 fail the complete-chain check. Support partial
range batches, or use a valid batch at the same science geometry and report its
effect. Do not change angles solely to make the scaling benchmark convenient.

Whole-scan distribution is the first throughput experiment. Dividing one volume
into independent z blocks across GPUs is a later option for large volumes or
latency targets; it adds distribution, reassembly and postprocessing work.
Cross-z Gaussian smoothing requires whole-volume handling or validated halos.
The persistent FBP prototype also uses shared geometry constants and a private
ASTRA ABI; concurrent geometries and multiple devices need explicit validation.

Double buffering is useful only if receiving/assembly can progress while the
other buffer is reconstructed. Merely allocating another buffer does not change
the synchronous production worker. The [combined local prototype](SCAN-OVERLAP-FOLLOWUP.md)
uses independent assembly and reconstruction workers and demonstrates GPU overlap,
but does not yet establish a reliable verified-throughput improvement. Existing
host block prefetch was observed not to overlap
ASTRA computation in the SIRT probe. Preserve stream completion and borrowed
buffer ownership in any new asynchronous path.

## Keep verification and memory honest

Disabling comparison identifies compute/transport capacity; it does not increase
verified throughput. An asynchronous verifier needs reusable buffers because
`DownloadedVolumes` overwrites its host array on the next read. Copying into an
unbounded queue would hide backpressure until memory runs out. A GPU verifier
must retain the independent reference and reproduce finite-value, tolerance and
phantom-error checks, including deliberate failure cases; on the reconstruction
GPU it may compete for compute and memory bandwidth. Fresh independent reference
generation for changing scans also needs capacity if it is a production
requirement. Sampling is a different verification policy and must be labelled.

Budget sinograms, reconstruction workspaces, output slots, GPU receive rings,
pinned pools, references and queued sink buffers per worker. The existing
four-volume output receive ring already consumes 512 MiB at the audited shape;
reducing it to two volumes caused severe waits in multi-worker tests. Larger
rings absorb bursts, not a sustained sink-capacity deficit. Host buffering and
block output are valuable capacity options but can add CPU memory, transfer and
assembly work; do not assume they improve throughput.

The source's required archive subscriber can backpressure the entire pipeline.
Keep archive completeness and required output semantics fixed across runs.
Measure writer bandwidth, archive bandwidth and CPU memory bandwidth once GPU
capacity increases; they can become the next limit. Allocate sufficient CPU
cores to progress, processing and verification, and avoid oversubscribed CPU
GridRec/BLAS threads. No current trace proves CPU or device-memory saturation.

## Define the rates before comparing them

Use the same elapsed interval T and distinguish these counters:

- **Verified volumes/s:** unique complete volumes accepted by the required
  verifier and output path, divided by T. Count only finished mandatory work;
  also report produced volumes/s and verification backlog separately.
- **Compressed input MB/s:** sum of actual compressed detector-frame payload
  bytes accepted during T, divided by T and 1,000,000. State whether dark/flat
  frames are included; the current source includes them.
- **Raw detector MB/s:** decompressed frames times rows times columns times
  detector element bytes, divided by T and 1,000,000.
- **Corrected projection MB/s:** useful float32 projections times rows times
  columns times 4, divided by T and 1,000,000.
- **Useful reconstructed output MB/s:** accepted complete volumes times rows
  times columns squared times 4, divided by T and 1,000,000. Exclude unused tail
  padding in fixed-capacity block messages; report wire bytes separately.

For 128 rows, 360 angles, 512 columns and uint16 detector data, one scan has
45.25 MiB of raw input including dark/flat, 90 MiB of corrected projections and
128 MiB of reconstructed output. Compressed size depends on the input and must
be counted. These are payload sizes, not predicted throughput rates.

Current `streaming_rates` reports **MiB/s** using 1,048,576 bytes/MiB and mean
rates since acquisition starts, including fill and drain. Its volume counter
also counts unverified runs. Convert MiB/s to decimal MB/s by multiplying by
1.048576 when needed. Do not sum stage payload rates: they describe different
representations of the same scans. Physical link traffic, additional subscribers
and internal copies require separate counters.

## Benchmark and decision rules

- Run a warmed, unprofiled sustained interval, with enough scans to expose a
  growing queue, plus a separate fill/drain-inclusive result. Keep source demand
  above achieved throughput for capacity tests and separately test the actual
  acquisition rate. Use varied representative inputs and production sinks.
- Alternate baseline/candidate order, repeat paired runs, and retain versions,
  device/HCA placement, geometry, filters, iterations, clocks/power state,
  batches, budgets, compressed byte counts and verification policy.
- Record accepted/produced volumes, each payload rate, scan latency, queue/ring
  occupancy, slot waits, per-stage service counters, memory peaks and errors.
  Combine these with CPU/GPU/memory/NIC/storage utilization on the target
  deployment. Do not add nested timings or times from concurrent processes.
- Use a short CUDA/NVTX capture and `UCX_PROTO_INFO=y` to confirm mechanisms,
  not to establish the final throughput. Retain cluster `ucx_info` capabilities
  and actual protocol selection; endpoint lane inventories are insufficient.
- Accept an improvement only when repeated verified-throughput gains exceed
  baseline variability, mandatory work finishes without loss, queues remain
  bounded and correctness remains valid. After each win, identify the new
  limiting stage before stacking another change. Do not multiply isolated gains.

Python records caching, discovery optimization, raising batches far beyond 16,
event-polling changes and arbitrary UCX segment overrides are lower priorities
for this workload. The native header helper measured about 26 ns/frame; receive
batching still leaves one active-message send per frame. Oversized SYSV segments
corrupt messages in the audited UCX build, and the tested rendezvous override
was not a working fix. Keep validated transport settings until a fixed build is
checked. No current evidence establishes a large default transport bottleneck.

## Evidence and code

Evidence: [follow-up summary](FOLLOWUP-SUMMARY.md), [audit trail](NOTES.md),
[targeted findings](TARGETED-FINDINGS.md), [FBP](FBP-FOLLOWUP.md),
[ingestion/layout](LAYOUT-FOLLOWUP.md), [metadata](METADATA-FOLLOWUP.md), and
[transport](TRANSPORT-FOLLOWUP.md), plus the [scan-overlap experiment](SCAN-OVERLAP-FOLLOWUP.md).

Code inspected: [worker dispatch](../../tomography/pipeline_device.cpp),
[processors](../../tomography/processors.py),
[placement, writer and rates](../../tomography/demo.py),
[ring sizing](../../tomography/pipeline_control.py), and
[memory planning](../../tomography/host_buffering.py).
This report summarizes local measurements and adds no production changes.
