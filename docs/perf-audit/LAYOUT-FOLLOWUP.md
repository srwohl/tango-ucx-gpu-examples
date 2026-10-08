# Projection-to-sinogram layout follow-up

Measured on 2026-10-07 with CuPy 14.2 and the local RTX 2060 SUPER.
The isolated `probes/layout_ingestion.py` compares the current strided
projection-to-sinogram writes, batched writes, and an additional contiguous
staging layout. It also measures the real `Processor.consume` path with its
validation and borrowed-pointer construction, stopping before the final
projection would trigger FBP.

## Results

Input is a GPU-resident, contiguous `(360, 128, 512)` float32 projection array.
Output is ASTRA's `(128, 360, 512)` sinogram layout. Arrays are preallocated;
initial uploads, reference checks and output downloads are outside timing.
Five repetitions follow a warmup:

| Operation | Copy calls | Median wall time per scan | Additional retained storage |
| --- | ---: | ---: | ---: |
| Current strided write per frame | 360 | 12.60 ms | 0 |
| Batched layout write, 16 frames | 23 | 0.84 ms | 0 |
| Batched layout write, 128 frames | 3 | 0.61 ms | 0 |
| One whole-scan layout write | 1 | 0.66 ms | 0 |
| Contiguous frame writes, then transpose copy | 361 | 7.57 ms | 90 MiB |
| Production `consume`, without final projection | 360 | 37.40 ms | 0 |
| Experimental validated grouped ingestion, 16 frames | 23 | 2.77 ms | 0 |
| Experimental validated grouped ingestion, 128 frames | 3 | 0.72 ms | 0 |

The input pattern varies over every axis; all variants exactly match the
transpose reference and leave input unchanged. Borrowed pointer views point
directly to the input allocation. Production ingestion receives 360 projections
of a declared 361-projection scan, and its untouched final sinogram plane remains
NaN. Its reconstruction counter remains zero, confirming that FBP is excluded.
Both ingestion paths use precomputed pointer and metadata tuples, matching the
C++ caller's pre-existing frame data rather than timing Python sender views.
The grouped path checks kinds, byte lengths, projection continuity, angles, and
contiguous frame addresses before constructing one borrowed view and enqueueing
the copy. An intentionally changed angle is rejected with no counter or output
mutation.

CUDA event intervals approximately follow the wall timings, but include gaps
while Python submits work. They are not isolated device kernel timings. These
results come from one isolated process with no concurrent agent GPU benchmark;
they do not establish a pipeline throughput gain.

## Interpretation

Batched writes can remove roughly 12 ms/scan of layout submission cost on
this machine. Compared with about 175 ms for isolated 128-slice FBP, this is a
secondary target. The full production ingestion loop is slower than the layout
copy loop because it also validates projection metadata and creates pointer
views per frame. The difference includes differing array strides (361 declared
angles in the production case) and cannot be attributed to one helper without
an additional focused measurement.

Changing the retained scan to projection-major order makes incoming writes
contiguous, but requires another 90 MiB allocation and a whole-scan transpose.
That is a weaker isolated result than using the existing transport batch to
write several sinogram projections together. There is no evidence here that a
host fallback or an accidental copy in `array_at` is the problem.

The batched-copy variants assume consecutive receive frames occupy one
contiguous allocation. A production implementation must split batches at ring
wraps and scan boundaries, validate all frame metadata before enqueueing work,
and retain the receive view until the stream has finished. The existing
transport batch lifetime can provide that hold, but the C++ pipeline currently
still invokes reconstruction ingestion frame by frame. Benchmarking a grouped
ingestion entry point is the next integration experiment; transport batch size
alone does not activate these copy variants. The isolated grouped prototype
reduces the measured ingestion envelope from 37.40 ms to 2.77 ms at batch 16,
while still validating the metadata. It deliberately stops before scan completion
and does not implement the C++ publication, calibration or scan-transition state
machine, so this is not yet a drop-in pipeline path.

## Reproduce

From `tango-ucx-gpu-examples/`:

```bash
pixi run python docs/perf-audit/probes/layout_ingestion.py \
  --output /tmp/layout-ingestion.json
```

Raw samples and validation details are in `layout-ingestion-grouped.json`;
`layout-ingestion.json` retains the earlier copy-only experiment. No production
memory-layout changes were made for this experiment.
