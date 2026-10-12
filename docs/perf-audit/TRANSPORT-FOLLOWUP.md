# Native transport follow-up

Measured 2026-10-07, using the current `tango-ucx` native transport static library,
UCX 1.22.0 (revision `8a6b06f`), two separate forked host processes, no GPU
allocation or compute. The probe is independent of Python receive wrappers and
Tango device discovery. Application source is unchanged.

## Main findings

1. The 256K/1M segment-size failure is reproduced with **raw UCX**, without any
   Tango publisher, subscription, wire schema or ring placement. It is a UCX
   shared-memory eager-message length overflow in this installed build.
2. Native binary header encoding/decoding costs about **26 ns/frame**, compared
   with the previously measured Python structured-dtype property construction
   of about 35 microseconds/access. Native metadata is not the same bottleneck.
3. Receive batches cut releases, reads and credit calls; the publisher still
   issues **one UCX active-message send per frame**. At 75 KB/frame the repeated
   transport envelope is about 32–45 microseconds/frame, with no consistent
   throughput improvement from raising receive batch 16 to 128.

## UCX segment failure: root cause and reproduction

`transport_native --raw` creates a plain `ucp_init` context with
`UCP_FEATURE_AM`, two separate workers, a whole-message AM handler, an 88-byte
header containing only `0xA5`, and a payload containing only `0x5A`. It forces
eager mode to reproduce the eager protocol actually selected by the native
publisher with large segments. This mode does not use Tango transport classes.

| SYSV segment | Sent payload bytes | Callback payload bytes | Header | Result |
| --- | ---: | ---: | --- | --- |
| 8256 | 75000 | 75000 | `0xA5` | Pass |
| 256K | 65000 | 65000 | `0xA5` | Pass |
| 256K | 65400 | 65400 | `0xA5` | Pass |
| 256K | 65535 | `18446744073709551615` | Corrupt | Fail |
| 256K | 65536 | 0 | `0x5A` | Fail |
| 256K | 65537 | 1 | `0x5A` | Fail |
| 256K | 66000 | 464 | `0x5A` | Fail |
| 256K | 75000 | 9464 | `0x5A` | Fail |
| 1M | 75000 | 9464 | `0x5A` | Fail |

The FIFO's length counts the full transport message, so the exact first failing
payload depends on headers and is below 65536. At 65535 the callback length
underflows to `SIZE_MAX`; the probe avoids dereferencing corrupted lengths.

Installed library debug types show `uct_mm_fifo_element.length` is `uint16_t`,
while `seg_size` is `size_t`: see `transport-uct-types.txt`. UCX 1.22.0's
[interface query](https://raw.githubusercontent.com/openucx/ucx/v1.22.0/src/uct/sm/mm/base/mm_iface.c)
advertises `am.max_bcopy = config.seg_size`; its
[send path](https://raw.githubusercontent.com/openucx/ucx/v1.22.0/src/uct/sm/mm/base/mm_ep.c)
assigns the packed `size_t` length to that FIFO length. These source operations,
the installed type and the raw callback results establish the overflow.

The current native publisher/subscriber reproduces the original application
error with both 256K and 1M: `the publisher sent an invalid frame`. A GDB capture
of the first rejected frame has `hn=88`, `n=9464`, a header full of payload bytes,
and credited/freed/arrived all zero (`transport-invalid-gdb.txt`). Validation
correctly rejects the malformed delivery before opaque-ring placement.

Default and 32K segment settings pass complete payload validation. These host
correctness probes do not establish a pipeline speedup for 32K. Setting the
documented `UCX_RNDV_THRESH=8K` with 256K removes the eager rejection in one
focused native run but delivers no frames before the 30-second read deadline;
outcome remains unset. That attempt is **not a validated workaround**. Evidence:
`transport-rndv-workaround.json`.

## Fixed metadata and actual transport envelope

Three repeats of 3 million helper encode/decode iterations give medians:
48-byte opaque header with no application fields: 27.53 ns/frame; 88-byte
opaque header with 40 application-field bytes: 25.70 ns/frame. These measure
the helper pair, not the complete publisher orchestration or allocations.

Each native transfer case sends 5000 frames, checks frame sequence, payload
length, first/last payload markers and application-field sequence, and reaches
`Outcome::End`. A 32 MiB publisher budget and 32 MiB subscriber budget are used;
receive max-wait is fixed at 10 ms. Payload fill and checks are included in the
reported interval; context creation, allocation and native Open/Ready setup
are excluded. Counts-only interception of `ucp_am_send_nbx` adds one relaxed
atomic operation per send. Three-repeat medians:

| Payload | Fields | Batch 1 | Batch 16 | Batch 128 |
| --- | ---: | ---: | ---: | ---: |
| 128 bytes | 0 | 2.90 us/frame | 1.77 | 1.64 |
| 128 bytes | 40 | 2.58 | 1.76 | 2.14 |
| 75000 bytes | 0 | 34.94 | 40.17 | 34.53 |
| 75000 bytes | 40 | 40.18 | 34.79 | 40.00 |

The 75 KB distributions overlap; these short local runs do not support a
precise metadata-field overhead or a batch-128 speedup. Small frames expose
per-read/release overhead more clearly. In all 36 successful steady-transfer
cases, publisher frame-send calls remain exactly 5000.

For 75 KB frames and 40-byte fields, observed credit-call ranges are
2727–3192 (batch 1), exactly 320 (batch 16), and 49–51 (batch 128).
Cumulative credits can already coalesce several scalar releases on a worker
pass; credits are not necessarily one per application read. Receive batches
are sometimes cut at packed-ring wraps, so batch 128 needs 51 reads, rather
than the ideal 40. The final credit snapshot may precede a last worker pass.

Optional host-call timing is separate from the repeated benchmark. Individual
75 KB diagnostic runs report 5000 frame calls and 3023/319/51 credits for
batches 1/16/128; summed credit-send host duration is below 0.90 ms each.
Frame-send host duration varies strongly with progress and shared-memory copy
work and is not interpreted as header cost. Source selection is one
`ucp_am_send_nbx` per frame; coalescing receiver batches cannot batch that send.

## Startup, discovery and protocol scope

Native publisher construction and subscriber preparation are each roughly
0.8–0.9 seconds in these fresh processes, including UCX context/worker setup,
transport discovery, registration/allocation and ring setup. Native Open
handling takes roughly 2–4 ms and Ready waits roughly 1 ms. No Tango command
discovery runs in this probe, so assigning this native startup envelope to
Tango metadata churn would be incorrect.

The previous Python probe used a staged extension that lacked current
`Batch.index`; this follow-up links the current rebuilt static transport
library, whose checksum is stored with the executable checksum in
`transport-matrix.json`. Fresh current native failures establish that segment
overflow also affects current source, not only that older staged extension.

`UCX_PROTO_INFO=y` explicitly selects `sysv/memory` host copy-in and multi-frag
copy-in in `transport-protocol-baseline.txt`. The endpoint health also lists
`cuda_ipc/cuda`, which is a lane inventory, not proof of GPU payload movement
in this host probe. Earlier actual GPU captures established local CUDA IPC;
this work measures no cluster RDMA or GPUDirect route.

On the cluster, retain `ucx_info -v`, `ucx_info -d` and `ucx_info -cf` alongside
one `UCX_PROTO_INFO=y` pipeline run with volume verification disabled. Inspect
the protocol selected for each source/receive memory-type pair and each link;
a list of endpoint lanes alone is insufficient. Check registered GPU-memory
rendezvous protocols and the intended HCA/port, and compare against an actual
acquisition trace. Hardware access remains necessary for that conclusion.

## Reproduce

From the directory that holds `tango-ucx` and `tango-ucx-gpu-examples`:

```bash
bash tango-ucx-gpu-examples/docs/perf-audit/probes/transport_build.sh
python3 tango-ucx-gpu-examples/docs/perf-audit/probes/transport_matrix.py --output /tmp/transport-matrix.json
UCX_SYSV_SEG_SIZE=256K tango-ucx-gpu-examples/docs/perf-audit/probes/transport_native --raw --bytes 75000
UCX_SYSV_SEG_SIZE=256K tango-ucx-gpu-examples/docs/perf-audit/probes/transport_native \
  --frames 100 --bytes 75000 --batch 16 --validate-all
tango-ucx-gpu-examples/docs/perf-audit/probes/transport_native --header --frames 3000000 --fields 40
```

The deliberate corrupt-delivery cases exit 2. The matrix records them as
expected diagnostic outcomes; any default-settings transfer failure makes
the matrix runner fail. `transport-varying.json` additionally validates every
payload byte for 5000 variable-length frames at default and 32K segments.

To reproduce the current native rejection inspection without editing source:

```bash
bash tango-ucx-gpu-examples/docs/perf-audit/probes/transport_debug_build.sh
UCX_SYSV_SEG_SIZE=256K gdb -q -batch \
  -x tango-ucx-gpu-examples/docs/perf-audit/probes/transport_invalid.gdb /tmp/transport-native-debug
```

The debug script compiles only the current subscriber translation unit with
symbols and links it ahead of the current optimized archive. That capture is
diagnostic and is excluded from performance measurements.

Next transport action: carry this standalone raw reproduction to the UCX
build/version used on the cluster and validate an upstream length clamp or
fixed build before changing segmentation settings. For application throughput,
native sender batching needs an explicit wire/send mechanism; current receiver
batching already makes native control cost small compared with the measured
FBP stage. No production transport fix is included in this follow-up.
