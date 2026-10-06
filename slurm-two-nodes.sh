#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --job-name=tango-ucx-gpu

set -euo pipefail
# Set EXAMPLES_ROOT explicitly when submitting: Slurm copies this script to a spool directory.
task_root=${EXAMPLES_ROOT:?Set EXAMPLES_ROOT to the shared absolute examples directory}
task_profile=${EXAMPLE_NETWORK:-rdma}
task_example=${EXAMPLE:-gpu_receive}
task_source=${EXAMPLE_SOURCE:-cuda:0}
task_python=${EXAMPLES_PYTHON:-python}
task_port=${EXAMPLES_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
task_results="$task_root/results/slurm-$SLURM_JOB_ID"
mkdir -p -- "$task_root/results"
mkdir -- "$task_results"
mapfile -t task_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if (( ${#task_nodes[@]} != 2 )); then
    echo "Allocate exactly two nodes" >&2
    exit 1
fi
task_network=(--profile "$task_profile")
if [[ -n ${EXAMPLES_NET_DEVICES:-} ]]; then
    task_network+=(--net-devices "$EXAMPLES_NET_DEVICES")
fi
task_device="${task_nodes[0]}:$task_port/example/opaque/1#dbase=no"

srun --exclusive --nodes=1 --ntasks=1 --nodelist="${task_nodes[0]}" \
    "$task_python" "$task_root/network.py" "${task_network[@]}" -- \
    "$task_root/build/opaque_publisher" hpc -nodb -dlist example/opaque/1 \
    -ORBendPoint "giop:tcp:${task_nodes[0]}:$task_port" \
    > "$task_results/publisher.log" 2>&1 &
task_server=$!
cleanup() {
    kill -TERM "$task_server" 2>/dev/null || true
    wait "$task_server" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

srun --exclusive --nodes=1 --ntasks=1 --nodelist="${task_nodes[1]}" \
    "$task_python" "$task_root/network.py" "${task_network[@]}" -- \
    "$task_python" "$task_root/hpc_subscriber.py" "$task_device" \
    --example "$task_example" --source "$task_source" --frames "${EXAMPLE_FRAMES:-1000}" \
    --access "${EXAMPLE_ACCESS:-pointer}" --output "$task_results/archive" \
    > "$task_results/subscriber.log" 2>&1
cat "$task_results/subscriber.log"
if [[ $task_example == host_archive ]]; then
    "$task_python" "$task_root/verify_archive.py" "$task_results/archive"
fi
