# Streamed dark and flat correction

The simulated streamer publishes typed `uint16` array frames on its UCX stream. For each
calibration ID it sends a stack of dark frames, a stack of flat frames, then data frames.
The next ID repeats that sequence with changed pixel offsets and gains. The shape is part
of the stream description; the record supplies `kind` (0 dark, 1 flat, 2 data),
`calibration_id` and `sample` within that phase. No Tango round trip is needed per frame.

`dark_flat.py` uses the installed `tango_ucx.every()` subscriber, its `Batch` and `GpuView`.
The default simulated shape is 129 × 131 with four dark frames and four flat frames per
calibration, and two calibrations in one run. `Start N` means N **data frames per
calibration**, plus the dark and flat frames. Every delivery ensures the subscriber sees
all the calibration frames. This example refuses latest and pull delivery: a calibration
distribution policy would be needed for subscribers that can skip or share these frames.

## GPU allocation and ownership

The application allocates five `float32` GPU arrays once: dark and flat accumulators,
dark and flat means, and the inverse response. Each new calibration resets the accumulators
and replaces the maps in these same allocations. All accumulation, averaging, map updates
and correction use the same caller-owned, non-default CUDA stream, so updates follow the
last data read of the preceding calibration without a per-frame synchronization.

These maps belong to the application. They never point into the receive ring and do not
keep calibration receive batches indefinitely. GPU receive imports a read-only DLPack
array within `batch.gpu_view(stream=...)`. Dark/flat pixels are accumulated into the owned
arrays, and data pixels are read by the correction kernel. The imported array is deleted
and the view ends after its final reads are queued; the library's completion event protects
the receive memory until those reads finish.

With `--receive pinned`, the application uploads into a bounded GPU input pool. It calls
`batch.release_after(stream=...)` immediately after the upload and drops the host batch
and NumPy array before correction. Analysis then reads its independent GPU allocation.
Outputs use a second bounded pool; a completion event prevents input/output slot reuse
until the queued kernels and verification finish. `--inflight` controls the number of
outstanding batches.

The simulated host publisher generates spatially varying offsets, gains and data on the
CPU. `--source cuda:0` also exercises real GPU source slots: it uploads the generated frame
to CUDA memory. That simulated source waits for upload completion before reusing its one
host generation buffer. GPU receive, calibration and correction remain real CUDA work;
there is no CPU fallback for the correction example.

## Correction convention and checks

The output is normalized intensity (transmission), using averaged calibration frames:

```text
dark = mean(dark frames)
flat = mean(flat frames)
corrected = (data - dark) / (flat - dark)
```

This is the flat/dark normalization convention described by
[TomoPy](https://tomopy.readthedocs.io/en/1.9.2/ipynb/tomopy.html). Flat frames here include
their dark offset; they are not precomputed reciprocal gain maps. A valid flat would
correct to one and a dark to zero. The output is `float32`, with no logarithm or clipping.
Subtracting in floating point preserves below-dark values instead of unsigned wraparound.

A flat-minus-dark response at or below `--minimum-response` (default 1) yields NaN, as does
saturated data (65,535 for these `uint16` frames). The simulation includes zero and negative
responses, saturated pixels, below-dark data, nonuniform offsets/gains and symmetric noise
in the calibration stacks. The new calibration changes both the offsets and response, so
using stale GPU maps fails the verification.

Every corrected pixel is compared with an independent NumPy float64 reference. Each
calibration's GPU means are compared exactly with the reference means, and the allocation
pointers are checked for reuse. The correction and inverse-response preparation run as
CuPy CUDA elementwise kernels. CPU work constructs the reference and checks results; it
does not supply GPU calibration maps or corrected data.

## Local TCP run

Build with `./build_local.sh`, then start the existing device server as described in the
[main README](README.md#run-locally-with-tcp). In another terminal using the same environment:

```sh
device='127.0.0.1:22080/example/opaque/1#dbase=no'
python control.py "$device" configure --kind calibration --source host \
  --shape 129,131 --dark-frames 4 --flat-frames 4 --calibrations 2
python network.py --profile tcp --net-devices lo -- \
  python dark_flat.py "$device" --receive gpu --start 23 --batch 5 \
  --inflight 3 --budget 1MiB --output results/calibration-gpu

python control.py "$device" configure --kind calibration --source host
python network.py --profile tcp --net-devices lo -- \
  python dark_flat.py "$device" --receive pinned --start 23 --batch 5 \
  --inflight 3 --budget 1MiB --output results/calibration-pinned
```

An output directory must be new. It receives `calibration-<id>.npz` with GPU-derived dark,
flat and inverse-response maps, plus `last_corrected.npy`. Correction outputs remain on
the GPU until this example reads them for verification or saving. The summary reports
calibration count, corrected data frames, verified pixels, maximum absolute error and
actual transports. `--expected` counts data frames across all calibrations.

The CTest entry `calibration_gpu_tcp` runs host and CUDA publishing into both GPU receive
and pinned upload: four cases of 62 total frames, including 46 corrected data frames each.
A batch of five crosses phase and calibration boundaries; the final batch is short. A
1MiB receive budget also exercises reuse while the persistent calibration stays on the GPU.
The test includes a fifth run through `hpc_subscriber.py`, using its configuration and
readiness sequence over local TCP.

## Validation on this machine

On the RTX 2050 with driver 580.178.04 and CUDA 12.9, all four source/receive combinations
and the local HPC subscriber launch passed. Each run corrected 46 data frames across two
calibration generations, checking all 777,354 output pixels against the float64 reference.
The maximum absolute error was below 2 × 10⁻¹¹. GPU dark and flat means matched exactly,
and the persistent map pointers stayed unchanged across the calibration update.

The complete examples suite passed all three CTest entries. The publisher also built with
CUDA disabled and passed its host archive test, both calibration receive paths and the
HPC subscriber launch with a real GPU on the receiving side. To repeat that arrangement
after building both configurations:

```sh
pixi run --manifest-path ../tango-bulk2/pixi.toml -e gpu \
  python smoke.py --publisher build-host/opaque_publisher --calibration-only --host-only
```

GPU processing requires actual hardware. RDMA transport and Slurm execution await the HPC
system.

## Two-node HPC run

Use the same TCP/RDMA profiles and installed-package setup as the other examples. On the
subscriber node, after starting the publisher on its node:

```sh
python network.py --profile rdma --net-devices mlx5_0:1 -- \
  python hpc_subscriber.py 'publisher-node:22080/example/opaque/1#dbase=no' \
  --example dark_flat --source host --shape 512,513 --frames 1000 \
  --receive gpu --batch 4 --inflight 4 --budget 32MiB --output results/calibration-hpc
```

Change `--receive` to `pinned` to compare uploads. Change both processes' profiles to `tcp`
and select their Ethernet interfaces to run without RDMA. The Slurm launcher also accepts:

```sh
sbatch --export=ALL,EXAMPLE=dark_flat,EXAMPLE_SOURCE=host,EXAMPLE_NETWORK=rdma slurm-two-nodes.sh
```

This single-subscriber launch delivers both calibration generations to that subscriber.
RDMA and Slurm execution still require the HPC system; successful cluster execution is not
established by the local TCP checks.
