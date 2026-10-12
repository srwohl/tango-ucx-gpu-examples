# Tomography performance probes

Runnable probes for GPU processing, reconstruction, buffer ownership and native
transport. Run commands from the repository root after building the examples:

```sh
pixi run build-pipeline
```

Write measurements to a fresh directory outside the repository. Probe results,
logs, images, detector data and profiler captures are generated locally and
ignored by Git.

| Measure | Probe |
| --- | --- |
| Decompression batch capacity | `probes/decompress_batches.py` |
| Correction, compression and target-size operations | `probes/target_scale.py`, `probes/compress_rates.py`, `probes/compress_typed.py` |
| FBP filtering and backprojection phases | `probes/fbp_phases.py`, `probes/fbp_split.py` |
| Projection-to-sinogram ingestion | `probes/layout_ingestion.py` |
| Python metadata access | `probes/metadata_access.py` |
| DLPack layouts and reconstruction handoffs | `probes/dlpack_probe.py`, `probes/astra_readonly.py`, `probes/handoff_probe.py` |
| Native FBP resource reuse | `probes/fbp_native.py`, `probes/fbp_pipeline.py` |
| Grouped assembly and scan overlap | `probes/scan_overlap_pipeline.py` |
| Native transport and header overhead | `probes/transport_build.sh`, `probes/transport_matrix.py` |
| TomocuPy backend comparisons | `probes/tomocupy_fbp.py`, `probes/tomocupy_pipeline.py` |
| Pipeline GPU utilization | `probes/gpu_saturation_run.sh` |

For example:

```sh
task_audit_output=$(mktemp -d /tmp/tomography-perf.XXXXXX)
pixi run python docs/perf-audit/probes/decompress_batches.py > "$task_audit_output/decompress.log"
pixi run python docs/perf-audit/probes/fbp_phases.py --output "$task_audit_output/fbp-phases.json"
pixi run python docs/perf-audit/probes/layout_ingestion.py --output "$task_audit_output/layout.json"
pixi run python docs/perf-audit/probes/metadata_access.py --output "$task_audit_output/metadata.json"
```

Pipeline comparison runners require a new output directory for each invocation:

```sh
pixi run python docs/perf-audit/probes/fbp_pipeline.py \
  --scans 10 --repeats 2 --verify --output /tmp/fbp-pipeline-comparison
pixi run python docs/perf-audit/probes/scan_overlap_pipeline.py \
  --backends production --with-baseline --verify --scans 10 --repeats 2 \
  --output /tmp/scan-overlap-comparison
```

The overlap probe can optionally record an earlier FBP comparison using
`--historical-results /path/to/results.json`; saved historical results are not
required to run it. Native FBP and the probe-specific TomocuPy build target
`sm_75`; the FBP integration shim uses GPU 0 and 128 slices, 360 angles and
512 detector columns. These probes use experimental backend interfaces.

The native transport builder expects a sibling `tango-ucx` checkout with its
GPU environment and `build-gpu` transport library already built:

```sh
bash docs/perf-audit/probes/transport_build.sh
pixi run python docs/perf-audit/probes/transport_matrix.py --output /tmp/transport-matrix.json
```

`probes/tomocupy/build.sh` builds the TomocuPy modules used by the comparison
probes. `pixi run build-tomocupy` builds the optional production backends.
Use a Pixi release that supports the checked-in build-tools lock files.

For CUDA timelines, run the selected probe under Nsight Systems with
`--trace=cuda,nvtx --sample=none --cpuctxsw=none` and put the capture in your
output directory. `UCX_PROTO_INFO=y` records the transport protocol selected for
each link in the run's device logs.
