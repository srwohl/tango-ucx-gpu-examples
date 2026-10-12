# Targeted bottleneck follow-up

Three new high-reasoning agents continued the FBP, metadata, and transport paths;
the root investigation covered layout and independently checked regressions.
Measurements below are local, on the RTX 2060 SUPER, not cluster/RDMA results.

| Path | Measured result | Status |
| --- | --- | --- |
| FBP | Paired full-pipeline acquisition mean 3.1692 → 2.6972 s: 14.9% less elapsed time, about 17% higher throughput. Isolated reconstruction improves 1.72×. | Experimental persistent native ASTRA path; no production backend replacement. |
| Metadata | Repeated Python records getter 37.18 → 1.49 µs/access. First access once per batch remains essentially unchanged. | Focused source fix with cached native records and independent read-only descriptors; not installed into examples/stage. |
| Layout | Isolated production per-frame ingestion 37.40 ms versus validated grouped ingestion 2.77 ms at batch 16. | Prototype, not a measured full-pipeline gain or a drop-in integration. |
| Scan overlap | Grouped asynchronous two-buffer assembly demonstrates GPU overlap. Current paired capacity improves about 3.6–3.8%; verified mean changes of 0.6%/3.0% remain within observed run variation. | Experiment-only generated device; no reliable verified-throughput win or production changes. |
| Transport | Native opaque encode/decode about 26 ns/frame; larger receive batches reduce credits but do not batch AM sends. | No transport production changes. Raw UCX reproduces oversized SYSV-segment corruption. |

## Evidence and limitations

- [FBP report](FBP-FOLLOWUP.md): scoped Nsight evidence, numerical comparisons,
  paired production/candidate runs, and independent normal volume verification.
  Persistent geometry removes staging allocations and recurring constant uploads.
  The true batched kernel is slower than persistent slice dispatch in these tests:
  fewer launches alone are not the solution. The candidate depends on ASTRA 2.5
  private ABI and serializes shared geometry constants; it needs hardening before
  production or multi-GPU use.
- [Metadata report](METADATA-FOLLOWUP.md): Python cache fix, first-access versus
  repeated-access benchmarks, lifetime and descriptor-mutation regressions.
  This improves Python consumers, not native C++ compute-stage metadata handling.
  The native dtype construction remains expensive on first access.
- [Layout report](LAYOUT-FOLLOWUP.md): borrowed-pointer checks, exact transpose
  validation, copy counts, and grouped-ingestion prototype. Production sinogram
  storage is strided for incoming projection slices; grouping avoids hundreds of
  small copy submissions without adding a second full-volume layout. Integration
  must split ring wraps and calibration/scan boundaries and preserve buffer lifetime.
- [Scan-overlap report](SCAN-OVERLAP-FOLLOWUP.md): combined grouped ingestion and
  independent assembly/reconstruction workers, historical and contemporaneous
  comparisons, normal per-volume verification, and direct next-scan copy/BP
  interval overlap. Two sinograms add 90 MiB at the audited shape. Historical
  baseline drift prevents attributing the slower historical comparison to the
  candidate; verified gains are below run variation.
- [Transport report](TRANSPORT-FOLLOWUP.md): native transfer matrix and raw UCX
  reproducer. UCX 1.22 advertises a larger SYSV bcopy capacity than its uint16
  receive-length field can represent. Tango correctly rejects malformed frames.
  Default settings pass; the tested rendezvous override stalls and is not a fix.
  Discovery/setup timings include allocation and registration, so they cannot be
  assigned to discovery alone. Cluster GPU zero-copy and RDMA remain unmeasured.

## Validation

The metadata change passes 41 host regressions, seven final GPU tests, and an
independent rerun of its three fixture regressions. FBP passes 48 numerical cases,
four ten-volume pipeline runs without failures/quarantine, and a separate
two-volume run with normal verification enabled. Transport passes 36 steady
transfer cases (180,000 frames), 10,000 variable-payload full-content checks, and
1,697 native assertions. Layout validates exact output/input preservation,
rejection before mutation, and an additional odd-size partial-batch case.
Scan overlap passes three focused GPU ownership/settings/error tests, forty
normally verified candidate volumes in matched comparisons, six initial verified
volumes, and a three-volume check with 97 projections and partial batches.

## Next targeted work

1. Harden the persistent FBP path behind an opt-in backend, keeping production
   filtering and independent volume verification; test multiple GPUs/geometries.
2. Keep the combined scan-overlap path experimental. Its local verified gains
   do not exceed run variation; grouped-only attribution and representative
   target-deployment checks remain before production integration.
3. Keep default UCX SYSV segment sizes until an upstream capacity clamp/fixed build
   is validated. Capture actual cluster protocol selection before tuning discovery
   or claiming end-to-end GPU zero-copy.

All probes, raw measurements, and trace commands are linked from the individual
reports. No commits were created and pre-existing demo edits were preserved.
