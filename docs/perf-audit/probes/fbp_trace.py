"""Collect bounded, scoped FBP evidence through the installed Nsight skill gateway."""
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

    evidence = {
        "doctor": run("report-doctor"),
        "kernel_fact": run("report-fact", "--type", "kernel_summary", "--max-rows", "5"),
        "api_fact": run("report-fact", "--type", "cuda_api_summary", "--max-rows", "5"),
    }
    scope = "n.text LIKE '%.rows128.parzen.repeat1'"
    runtime_join = ('FROM NVTX_EVENTS n JOIN CUPTI_ACTIVITY_KIND_RUNTIME a '
        'ON a.globalTid = n.globalTid AND a.start >= n.start AND a."end" <= n."end" ')
    queries = {
        "scope_wall": f'SELECT n.text AS phase, (n."end"-n.start)/1e6 AS wall_ms '
            f'FROM NVTX_EVENTS n WHERE {scope} ORDER BY phase LIMIT 10',
        "scoped_apis": 'SELECT n.text AS phase, names.value AS api, COUNT(*) AS calls, '
            'SUM(a."end"-a.start)/1e6 AS host_ms ' + runtime_join +
            f'JOIN StringIds names ON names.id=a.nameId WHERE {scope} '
            'GROUP BY n.text, names.value ORDER BY phase, host_ms DESC LIMIT 100',
        "scoped_bp_kernels": 'SELECT n.text AS phase, names.value AS kernel, COUNT(*) AS launches, '
            'SUM(k."end"-k.start)/1e6 AS gpu_ms ' + runtime_join +
            'JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=a.correlationId '
            'AND k.globalPid=((a.globalTid >> 24) << 24) '
            f'JOIN StringIds names ON names.id=k.shortName WHERE {scope} '
            "AND (names.value LIKE '%devBP%' OR names.value LIKE '%batch_backproject%') "
            'GROUP BY n.text, names.value ORDER BY phase LIMIT 10',
        "scoped_copies": 'SELECT n.text AS phase, m.copyKind AS copy_kind, COUNT(*) AS copies, '
            'SUM(m.bytes) AS bytes, SUM(m."end"-m.start)/1e6 AS gpu_ms ' + runtime_join +
            'JOIN CUPTI_ACTIVITY_KIND_MEMCPY m ON m.correlationId=a.correlationId '
            f'AND m.globalPid=((a.globalTid >> 24) << 24) WHERE {scope} '
            'GROUP BY n.text, m.copyKind ORDER BY phase, bytes DESC LIMIT 20',
    }
    for name, query in queries.items():
        evidence[name] = run("report-query", "--sql", query)
        print(json.dumps({name: evidence[name]["data"].get("rows", [])}), flush=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
