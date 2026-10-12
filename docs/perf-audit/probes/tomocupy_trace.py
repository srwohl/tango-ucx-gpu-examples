"""Collect scoped stream, copy and kernel evidence for the TomocuPy spike through the Nsight skill."""
import argparse
import json
from pathlib import Path
import subprocess

NSYS_PYTHON = "/opt/nvidia/nsight-systems/2026.5.1/target-linux-x64/python/bin/python"
NSYS_SKILL = "/opt/nvidia/nsight-systems/2026.5.1/skills/nsight-systems/scripts/nsys_skill_cli.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def run(command, *options):
        result = subprocess.run([NSYS_PYTHON, NSYS_SKILL, command, "--report", str(args.report),
                                 *options], check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        if not data.get("ok"):
            raise RuntimeError(result.stdout)
        return data

    evidence = {"doctor": run("report-doctor")}
    # The timed repeat of every method, and the two kernels that label the streams.
    scope = "(n.text LIKE '%.repeat1' OR n.text LIKE 'marker.%')"
    runtime_join = ('FROM NVTX_EVENTS n JOIN CUPTI_ACTIVITY_KIND_RUNTIME a '
        'ON a.globalTid = n.globalTid AND a.start >= n.start AND a."end" <= n."end" ')
    kernel_join = (runtime_join + 'JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=a.correlationId '
        'AND k.globalPid=((a.globalTid >> 24) << 24) ')
    copy_join = (runtime_join + 'JOIN CUPTI_ACTIVITY_KIND_MEMCPY m ON m.correlationId=a.correlationId '
        'AND m.globalPid=((a.globalTid >> 24) << 24) '
        'JOIN ENUM_CUDA_MEMCPY_OPER kinds ON kinds.id=m.copyKind ')
    own = "(n.text LIKE 'tomocupy%.repeat1' OR n.text LIKE 'marker.%')"
    queries = {
        "scope_wall": f'SELECT n.text AS phase, (n."end"-n.start)/1e6 AS wall_ms '
            f'FROM NVTX_EVENTS n WHERE {scope} ORDER BY n.start LIMIT 20',
        "scoped_streams": 'SELECT n.text AS phase, COUNT(DISTINCT k.streamId) AS streams, '
            'COUNT(*) AS launches, SUM(k."end"-k.start)/1e6 AS gpu_ms ' + kernel_join +
            f'WHERE {scope} GROUP BY n.text ORDER BY phase LIMIT 20',
        "tomocupy_kernel_streams": 'SELECT n.text AS phase, k.streamId AS stream, '
            'COUNT(*) AS launches, SUM(k."end"-k.start)/1e6 AS gpu_ms ' + kernel_join +
            f'WHERE {own} GROUP BY n.text, k.streamId ORDER BY phase, launches DESC LIMIT 50',
        "scoped_copies": 'SELECT n.text AS phase, kinds.name AS copy_kind, COUNT(*) AS copies, '
            'SUM(m.bytes) AS bytes, SUM(m."end"-m.start)/1e6 AS gpu_ms ' + copy_join +
            f'WHERE {scope} GROUP BY n.text, kinds.name ORDER BY phase, bytes DESC LIMIT 50',
        "tomocupy_copy_streams": 'SELECT n.text AS phase, kinds.name AS copy_kind, '
            'm.streamId AS stream, COUNT(*) AS copies, SUM(m.bytes) AS bytes ' + copy_join +
            f'WHERE {own} GROUP BY n.text, kinds.name, m.streamId ORDER BY phase, bytes DESC LIMIT 50',
        "scoped_apis": 'SELECT n.text AS phase, names.value AS api, COUNT(*) AS calls, '
            'SUM(a."end"-a.start)/1e6 AS host_ms ' + runtime_join +
            f'JOIN StringIds names ON names.id=a.nameId WHERE {scope} '
            'GROUP BY n.text, names.value ORDER BY phase, host_ms DESC LIMIT 200',
        "scoped_kernels": 'SELECT n.text AS phase, names.value AS kernel, COUNT(*) AS launches, '
            'SUM(k."end"-k.start)/1e6 AS gpu_ms ' + kernel_join +
            f'JOIN StringIds names ON names.id=k.shortName WHERE {scope} '
            'GROUP BY n.text, names.value ORDER BY phase, gpu_ms DESC LIMIT 200',
    }
    for name, query in queries.items():
        evidence[name] = run("report-query", "--sql", query)
        print(json.dumps({name: evidence[name]["data"].get("rows", [])}), flush=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
