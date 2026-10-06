# tango-ucx GPU examples

A separate application project for the GPU interfaces added to `tango-bulk2`. It builds
against the installed CMake package and imports the installed Python package. It includes
no library-private headers, fixtures or decoder code. The publisher and subscribers run
in separate processes and can run on separate nodes. No library count limits change here.

| Example | Path exercised |
|---|---|
| `gpu_receive.py` | Opaque GPU receive, per-frame lengths, DLPack 1.0 and raw GPU pointers; CuPy verifies every byte on an explicit non-default CUDA stream |
| `pinned_upload.py` | Page-locked host receive, asynchronous upload and `release_after`; GPU analysis reads independent output storage after receive Credit can return |
| `host_archive.py` | GPU source to host receive; stores opaque bytes and an index with the actual lengths and records; `verify_archive.py` reads the files back |
| `dark_flat.py` | Streamed dark/flat calibration and GPU correction of data frames, using the established subscriber; see [the calibration example](CALIBRATION.md) |
| `tomography/` | Four Tango device servers: compressed source → GPU LZ4 decompression → GPU correction → selectable GridRec (CPU), FBP (GPU) or SIRT (GPU) reconstruction, with continuous replay, concurrent compressed archival and a live browser view; see [the end-to-end pipeline](tomography/README.md) |

`opaque_publisher` is a C++ Tango device server. Configure selects `host` or `cuda:N` source
storage. The CUDA path queues real `cudaMemsetAsync` writes directly into source slots and
publishes without a caller synchronization. The application supplies each actual byte
length to `publish_bytes`. The default lengths, 1, 129, 513, 65,539 and 262,144 bytes, exercise
unaligned pointers, small messages, rendezvous and packed-ring wrap. Each frame is filled
with `(index * 17 + 3) & 255`; its record carries the value and length. These are generated
bytes for testing transport and ownership. They are not compressed detector frames.

## Build against the installed package

From this directory on the development machine:

```sh
./build_local.sh
```

This builds the sibling `tango-bulk2` CUDA package in its existing Pixi GPU environment,
installs it into `stage/`, and builds this independent project using `find_package`.
It runs three CTest entries: host TCP, GPU TCP and streamed GPU calibration. The opaque
pipeline cases use 97 frames and a 4MiB receive budget. The archive checks independently
reopen both host- and GPU-published files. Calibration checks cover host and GPU publishing
into GPU and pinned receive, with two calibration generations per run. Both kinds of
subscriber also run through the HPC launch sequence locally against the separate publisher.
Build output and dependencies belong to this directory; all library changes remain in the
library tree. There is no simulated GPU path.

For an installed HPC package, activate the site's Tango/UCX/CUDA/Python environment and use:

```sh
cmake -GNinja -S . -B build -DCMAKE_PREFIX_PATH=/path/to/tango-ucx-prefix \
  -DEXAMPLES_PYTHON_PREFIX=/path/to/tango-ucx-prefix -DEXAMPLES_CUDA=ON
cmake --build build
TANGO_UCX_PREFIX=/path/to/tango-ucx-prefix ctest --test-dir build --output-on-failure
```

The environment needs UCX >=1.21 with CUDA support, cppTango and PyTango >=10, a C++20
compiler, nlohmann_json >=3.12, NumPy and CuPy >=14. The installed package's CMake config
resolves the C++ dependencies. Use the Python version that built its extension. The source
publisher needs the CUDA toolkit/runtime but does not need nvcc: it calls the runtime.
Logical GPU numbers are relative to each process's `CUDA_VISIBLE_DEVICES`.

A host-only build is available with `EXAMPLES_CUDA=OFF`, against a host-only tango-ucx
installation; `./build_host.sh` builds and tests it locally in the default Pixi environment.
Its smoke test uses `--host-only`, and neither NumPy verification nor the host
archive imports CuPy. GPU examples still require real hardware.

## Select the network on both processes

`network.py` sets UCX policy before the program starts, including the library-specific
`TANGO_UCX_UCX_*` prefix. Pass it on the publisher node and on every subscriber node.

| Profile | `UCX_TLS` | Behavior |
|---|---|---|
| `tcp` | `tcp,cuda_copy,self` | Forces network TCP, including on one machine; CUDA storage and kernels still use the actual GPU. Enables the library's GPU-over-TCP opt-in |
| `rdma` | `rc,cuda` | Selects RC and CUDA support, excludes TCP and shared-memory shortcuts, and disables the TCP opt-in. The subscriber requires a negotiated RC/DC transport |
| `auto` | `all` | Lets UCX choose; permits TCP and reports the chosen transports |

`--net-devices` selects the local Ethernet interface or RDMA HCA and port. It can differ
between nodes. Without this flag, the existing UCX environment or UCX's normal selection
applies. Other UCX tuning settings remain available, including `UCX_PROTO_INFO=y` and
`UCX_LOG_LEVEL=info`. Profiles and device selection follow the
[UCX transport configuration documentation](https://openucx.readthedocs.io/en/master/faq.html#which-transports-does-ucx-use).

An RDMA transport in health does not establish GPUDirect RDMA. UCX can stage GPU data through
host memory; direct GPU registration depends on the cluster's drivers and peer-memory or
DMA-BUF support. Inspect protocol output on two different nodes to distinguish those paths.
See [UCX GPU support and RDMA requirements](https://openucx.readthedocs.io/en/master/faq.html#does-ucx-support-zero-copy-for-gpu-memory-over-rdma).

## Run locally with TCP

Activate the development CUDA environment, or use `pixi run --manifest-path
../tango-bulk2/pixi.toml -e gpu` before the commands below. Run the publisher in one terminal:

```sh
python network.py --profile tcp --net-devices lo -- \
  ./build/opaque_publisher local -nodb -dlist example/opaque/1 \
  -ORBendPoint giop:tcp:127.0.0.1:22080
```

In another terminal:

```sh
device='127.0.0.1:22080/example/opaque/1#dbase=no'

python control.py "$device" configure --source host
python network.py --profile tcp --net-devices lo -- \
  python gpu_receive.py "$device" --start 97 --access dlpack

python control.py "$device" configure --source host
python network.py --profile tcp --net-devices lo -- \
  python gpu_receive.py "$device" --start 97 --access pointer

python control.py "$device" configure --source host
python network.py --profile tcp --net-devices lo -- \
  python pinned_upload.py "$device" --start 97

python control.py "$device" configure --source cuda:0
python network.py --profile tcp --net-devices lo -- \
  python host_archive.py "$device" --start 97 --output results/gpu-archive
python verify_archive.py results/gpu-archive
```

Archive directories must be new; existing directories are refused. The same GPU source can
also feed `gpu_receive.py` or `pinned_upload.py`. Both GPU subscribers allow several batches
in flight (`--inflight 4` by default). Raw pointer reads must be queued before the GPU view
ends. DLPack tensors retain their receive batch until deleted. Pinned receive drops the
batch and its NumPy views after recording upload completion; its output allocations remain
alive through analysis completion.

## Run on separate HPC nodes

Build and install in a shared directory, load the same software environment on both nodes,
and allow Tango's TCP control traffic between them. UCX bulk transfer uses the selected
profile independently of Tango control. A no-database Tango address avoids a Tango database.

Publisher node, replacing the hostname and HCA with that node's actual names:

```sh
python network.py --profile rdma --net-devices mlx5_0:1 -- \
  ./build/opaque_publisher hpc -nodb -dlist example/opaque/1 \
  -ORBendPoint giop:tcp:publisher-node:22080
```

Subscriber node:

```sh
python network.py --profile rdma --net-devices mlx5_0:1 -- \
  python hpc_subscriber.py 'publisher-node:22080/example/opaque/1#dbase=no' \
  --source cuda:0 --example gpu_receive --access pointer --frames 1000
```

`hpc_subscriber.py` waits for the publisher, configures a fresh instance, joins a
subscription and starts it. Change `--example` to `pinned_upload` or `host_archive` for the
other paths. Change both profiles to `tcp` and use the nodes' Ethernet interfaces to run
without RDMA. No SSH access or MPI launcher is required by the programs themselves.

For Slurm, `slurm-two-nodes.sh` starts a publisher on the first node and a subscriber on
the second. Set a shared absolute examples path; add your site's account and partition
options to `sbatch`. Use a Python/environment already available on compute nodes:

```sh
export EXAMPLES_ROOT="$PWD"
export TANGO_UCX_PREFIX="$PWD/stage"
export EXAMPLES_NET_DEVICES=mlx5_0:1  # omit to let UCX select; this script uses it on both nodes
sbatch --export=ALL,EXAMPLE_NETWORK=rdma,EXAMPLE=gpu_receive,EXAMPLE_SOURCE=cuda:0 slurm-two-nodes.sh
sbatch --export=ALL,EXAMPLE_NETWORK=rdma,EXAMPLE=pinned_upload,EXAMPLE_SOURCE=host slurm-two-nodes.sh
sbatch --export=ALL,EXAMPLE_NETWORK=rdma,EXAMPLE=host_archive,EXAMPLE_SOURCE=cuda:0 slurm-two-nodes.sh
# Select an Ethernet interface, or unset EXAMPLES_NET_DEVICES, before the TCP comparison.
unset EXAMPLES_NET_DEVICES
sbatch --export=ALL,EXAMPLE_NETWORK=tcp,EXAMPLE=gpu_receive,EXAMPLE_SOURCE=cuda:0 slurm-two-nodes.sh
```

Results go into `results/slurm-<job-id>/`, with publisher and subscriber logs. The subscriber
reports actual transports, verified frames, payload bytes, skips and elapsed time. Time
includes Python reductions and verification; this is a correctness example, not a
transport throughput benchmark. `--expected N` checks an exact count where useful.

The opaque examples support every, latest and pull delivery. For several subscribers, run Configure
once, start each subscriber without `--start`, wait for each `{"ready": true, ...}` line,
then call `python control.py "$device" start N`. Pullers need the same `--range`, a whole
number of batches; together their frame indices should cover the run. Only one process
should initiate Configure or Start. `hpc_subscriber.py` is the single-subscriber shortcut.
The streamed calibration example requires every delivery to receive all dark and flat frames.

## Validation on this machine

The installed CUDA package built and the separate-process TCP smoke checks passed on the
RTX 2050, driver 580.178.04 and CUDA 12.9. Checks cover host source to GPU via both interfaces,
host source to pinned upload, GPU source to host archive, GPU source to GPU pointer analysis,
and GPU source to pinned upload. A separate host-only build checks host publishing and
archive verification without CUDA or CuPy. RDMA and Slurm execution require the HPC system;
this machine has no RDMA HCA. The RDMA profile was also tried locally: UCX refused startup
with `rc` unavailable, and did not fall back to TCP. Slurm script syntax passes `bash -n`;
the HPC subscriber's launch sequence passes against the local publisher. The calibration
checks also passed with real GPU-owned maps and correction through both receive paths;
see [calibration validation](CALIBRATION.md#validation-on-this-machine).
