# tango-ucx GPU examples

Runnable Tango publishers and Python subscribers for moving opaque frames into GPU,
pinned host and ordinary host memory. The programs use the installed
[tango-ucx](https://github.com/srwohl/tango-ucx) C++ and Python packages and run in
separate processes, locally or on separate nodes.

Use these examples to integrate CUDA stream handling, borrowed receive buffers,
asynchronous uploads and archival into an application. They verify payloads and
ownership through the public API. The tomography example adds an end-to-end processing
pipeline with compressed archival and a live browser view.

## Choose an example

| Example | What it demonstrates |
|---|---|
| [gpu_receive.py](gpu_receive.py) | Opaque GPU receive through DLPack or raw GPU pointers, with CuPy verification on an explicit CUDA stream |
| [pinned_upload.py](pinned_upload.py) | Pinned host receive, asynchronous upload and `release_after`; analysis continues in independent GPU storage |
| [host_archive.py](host_archive.py) | Host archival of opaque bytes, per-frame lengths and records, including frames published from GPU memory |
| [dark_flat.py](dark_flat.py) | Streamed dark/flat calibration and GPU correction; see [CALIBRATION.md](CALIBRATION.md) |
| [tomography](tomography/README.md) | GPU LZ4 decompression and correction, selectable reconstruction, continuous replay, archival and live viewing |

`opaque_publisher` is a C++ Tango device server with `host` and `cuda:N` source storage.
It generates test bytes using `(index * 17 + 3) & 255`, with the value and actual length
in each record. The default lengths are 1, 129, 513, 65,539 and 262,144 bytes to exercise
unaligned access, small messages, rendezvous and packed-ring wrap. For compressed
detector scans, use the tomography example.

## Requirements and build

Use Linux with Pixi, UCX >=1.21 with CUDA support, cppTango/PyTango >=10, a C++20
compiler, nlohmann_json >=3.12, NumPy and CuPy >=14. GPU paths require an NVIDIA GPU
and a driver compatible with CUDA 12.9. Use the Python version that built the installed
extension. GPU numbers are relative to each process's `CUDA_VISIBLE_DEVICES`.

From this directory, point the build script at the library checkout:

```sh
export TANGO_UCX_SOURCE=/path/to/tango-ucx
./build_local.sh
```

The script builds the library in its Pixi GPU environment, installs it into `stage/`,
builds these examples in `build/`, and runs host TCP, GPU TCP and calibration checks.
A sibling checkout can be selected with `TANGO_UCX_SOURCE=../tango-ucx`.

For host publishing and archival without CUDA or CuPy:

```sh
TANGO_UCX_SOURCE=/path/to/tango-ucx ./build_host.sh
```

This uses the library's default environment and creates `stage-host/` and `build-host/`.

To build against an existing installation, activate its Tango/UCX/CUDA/Python environment:

```sh
cmake -GNinja -S . -B build -DCMAKE_PREFIX_PATH=/path/to/tango-ucx-prefix \
  -DEXAMPLES_PYTHON_PREFIX=/path/to/tango-ucx-prefix -DEXAMPLES_CUDA=ON
cmake --build build
TANGO_UCX_PREFIX=/path/to/tango-ucx-prefix ctest --test-dir build --output-on-failure
```

Use `EXAMPLES_CUDA=OFF` for a host-only installation. The publisher calls the CUDA
runtime directly; its build does not require nvcc.

## Run locally

After `build_local.sh`, activate the library's GPU environment in each terminal:

```sh
pixi shell --manifest-path /path/to/tango-ucx/pixi.toml -e gpu
```

From the examples directory, start the publisher in one terminal:

```sh
python network.py --profile tcp --net-devices lo -- \
  ./build/opaque_publisher local -nodb -dlist example/opaque/1 \
  -ORBendPoint giop:tcp:127.0.0.1:22080
```

In another terminal, configure host publishing and receive 97 frames into GPU memory:

```sh
device='127.0.0.1:22080/example/opaque/1#dbase=no'
python control.py "$device" configure --source host
python network.py --profile tcp --net-devices lo -- \
  python gpu_receive.py "$device" --start 97 --access dlpack
```

Configure a fresh publisher instance before each run. To use GPU pointers, change
`--access dlpack` to `--access pointer`. To upload from pinned host receive:

```sh
python control.py "$device" configure --source host
python network.py --profile tcp --net-devices lo -- \
  python pinned_upload.py "$device" --start 97
```

To archive frames published from GPU source slots:

```sh
python control.py "$device" configure --source cuda:0
python network.py --profile tcp --net-devices lo -- \
  python host_archive.py "$device" --start 97 --output results/gpu-archive
python verify_archive.py results/gpu-archive
```

Archive output directories must be new. For host-only runs, use `build-host/opaque_publisher`
and `network.py --prefix ./stage-host` in the library's default environment.

### Ownership and multiple subscribers

GPU subscribers allow several batches in flight (`--inflight 4` by default). Queue raw
pointer reads before the GPU view ends. DLPack tensors retain their receive batch until
released. Pinned receive records upload completion, then releases the batch and its
NumPy views; independent output storage stays alive through analysis completion.

The opaque examples accept `--delivery every|latest|pull`. For several subscribers,
configure once, start each without `--start`, wait for every `{"ready": true, ...}` line,
then run `python control.py "$device" start N`. Pullers must use the same `--range`,
a whole number of batches. Only one process should configure or start the publisher.
The calibration example uses every delivery so it receives all dark and flat frames.

## Select the network

Wrap the publisher and every subscriber with `network.py`. It sets UCX policy before
the process starts, including the library-specific `TANGO_UCX_UCX_*` settings, and
adds the installed Python package from `--prefix` or `TANGO_UCX_PREFIX` (default `stage/`).

| Profile | `UCX_TLS` | Use |
|---|---|---|
| `tcp` | `tcp,cuda_copy,self` | TCP comparisons, including on one host; GPU work uses real hardware and GPU-over-TCP receive is enabled |
| `rdma` | `rc,cuda` | RC/CUDA transport without TCP or shared-memory shortcuts; subscribers require a negotiated RC/DC transport |
| `auto` | `all` | Let UCX choose and report the negotiated transports |

`--net-devices` selects the local Ethernet interface or HCA and port; names can differ
between nodes. If omitted, the existing UCX environment/default selection applies.
Use `UCX_PROTO_INFO=y` or `UCX_LOG_LEVEL=info` to inspect the chosen path.
An RDMA transport alone does not establish GPUDirect RDMA: GPU registration depends
on the target drivers and peer-memory or DMA-BUF support, and UCX can stage through
host memory. See [UCX transport configuration](https://openucx.readthedocs.io/en/master/faq.html#which-transports-does-ucx-use)
and [GPU RDMA requirements](https://openucx.readthedocs.io/en/master/faq.html#does-ucx-support-zero-copy-for-gpu-memory-over-rdma).

## Run on separate nodes

Install in a shared directory, load the same software environment on both nodes, and
allow Tango TCP control traffic between them. UCX payload transfer uses the selected
profile. The examples use no-database Tango addresses.

On the publisher node, replace the hostname and HCA with local values:

```sh
python network.py --profile rdma --net-devices mlx5_0:1 -- \
  ./build/opaque_publisher hpc -nodb -dlist example/opaque/1 \
  -ORBendPoint giop:tcp:publisher-node:22080
```

On the subscriber node:

```sh
python network.py --profile rdma --net-devices mlx5_0:1 -- \
  python hpc_subscriber.py 'publisher-node:22080/example/opaque/1#dbase=no' \
  --source cuda:0 --example gpu_receive --access pointer --frames 1000
```

`hpc_subscriber.py` waits for the publisher, configures it, joins and starts acquisition.
Choose `--example pinned_upload` or `--example host_archive` for the other receive paths.
For TCP, change both profiles and select each node's Ethernet interface. RDMA needs
suitable hardware on the target nodes.

### Slurm

`slurm-two-nodes.sh` starts a publisher on the first node and a subscriber on the second.
Set a shared absolute examples path and add the site's account and partition options:

```sh
export EXAMPLES_ROOT="$PWD"
export TANGO_UCX_PREFIX="$PWD/stage"
export EXAMPLES_NET_DEVICES=mlx5_0:1
sbatch --export=ALL,EXAMPLE_NETWORK=rdma,EXAMPLE=gpu_receive,EXAMPLE_SOURCE=cuda:0 slurm-two-nodes.sh
sbatch --export=ALL,EXAMPLE_NETWORK=rdma,EXAMPLE=pinned_upload,EXAMPLE_SOURCE=host slurm-two-nodes.sh
sbatch --export=ALL,EXAMPLE_NETWORK=rdma,EXAMPLE=host_archive,EXAMPLE_SOURCE=cuda:0 slurm-two-nodes.sh
```

Omit `EXAMPLES_NET_DEVICES` to let UCX select. The launcher uses the same selector on
both nodes. For TCP comparisons, choose an Ethernet interface or unset that variable,
and set `EXAMPLE_NETWORK=tcp`.

Results are saved in `results/slurm-<job-id>/` with both process logs. Subscribers
report transports, verified frames, bytes, skips and elapsed time; `--expected N`
checks an exact frame count. Timing includes Python analysis and verification. Use
[tango-ucx-benchmarks](https://github.com/srwohl/tango-ucx-benchmarks) for transport measurements.
