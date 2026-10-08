"""Attribute grouped assembly kernels and check overlap with the preceding scan's BP."""
import argparse
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    scope = "ranges.text LIKE 'overlap.%'"
    runtime_join = ('FROM NVTX_EVENTS ranges JOIN CUPTI_ACTIVITY_KIND_RUNTIME runtime '
                    'ON runtime.globalTid=ranges.globalTid AND runtime.start>=ranges.start '
                    'AND runtime."end"<=ranges."end" ')
    scoped = ('SELECT ranges.text AS phase, ranges.eventId AS range_id, kernel.globalPid AS process_tag, '
              'CAST(substr(ranges.text,instr(ranges.text,\'scan\')+4) AS INTEGER) AS scan_id, '
              'kernel.start AS start, kernel."end" AS stop, kernel.streamId AS stream, '
              'kernel.contextId AS context, names.value AS name ' + runtime_join +
              'JOIN CUPTI_ACTIVITY_KIND_KERNEL kernel ON kernel.correlationId=runtime.correlationId '
              'AND kernel.globalPid=((runtime.globalTid >> 24) << 24) '
              f'JOIN StringIds names ON names.id=kernel.shortName WHERE {scope}')
    connection.execute("CREATE TEMP TABLE scoped AS " + scoped)
    connection.execute("CREATE INDEX temp.scoped_start ON scoped(process_tag,start)")
    connection.execute("CREATE INDEX temp.scoped_stop ON scoped(process_tag,stop)")
    connection.execute("CREATE TEMP TABLE assembly AS SELECT * FROM scoped WHERE phase LIKE 'overlap.assemble.%'")
    connection.execute("CREATE TEMP TABLE bp AS SELECT * FROM scoped WHERE phase LIKE 'overlap.reconstruct.%' AND name LIKE '%devBP%'")
    connection.execute("CREATE INDEX temp.bp_timing ON bp(process_tag,scan_id,start)")
    connection.execute('CREATE TEMP TABLE runtime_correlations AS SELECT DISTINCT correlationId, '
                       '((globalTid >> 24) << 24) AS process_tag FROM CUPTI_ACTIVITY_KIND_RUNTIME '
                       'WHERE ((globalTid >> 24) << 24) IN (SELECT DISTINCT process_tag FROM scoped)')
    connection.execute("CREATE UNIQUE INDEX temp.runtime_correlation ON runtime_correlations(process_tag,correlationId)")
    overlap_join = ("FROM assembly CROSS JOIN bp ON assembly.process_tag=bp.process_tag "
                    "AND assembly.context=bp.context AND assembly.stream!=bp.stream "
                    "AND assembly.start<bp.stop AND bp.start<assembly.stop "
                    "AND assembly.scan_id=bp.scan_id+1 "
                    "WHERE assembly.phase LIKE 'overlap.assemble.%' "
                    "AND bp.phase LIKE 'overlap.reconstruct.%' AND bp.name LIKE '%devBP%'")
    queries = {
        "phase_host": ('SELECT ((globalTid >> 24) << 24) AS process_tag,text AS phase,COUNT(*) AS ranges, '
                       'SUM("end"-start)/1e6 AS host_ms FROM NVTX_EVENTS WHERE text LIKE \'overlap.%\' '
                       'GROUP BY process_tag,text ORDER BY process_tag,text LIMIT 40'),
        "scoped_kernels": ("SELECT process_tag,CASE WHEN phase LIKE 'overlap.assemble.%' THEN 'assembly' "
                           "ELSE 'reconstruction' END AS operation,name,COUNT(*) AS launches, "
                           "SUM(stop-start)/1e6 AS gpu_ms FROM scoped GROUP BY process_tag,operation,name "
                           "ORDER BY process_tag,operation,gpu_ms DESC LIMIT 40"),
        "gpu_overlap": ('SELECT assembly.process_tag,COUNT(*) AS overlapping_kernel_pairs, '
                        'COUNT(DISTINCT assembly.range_id) AS assembly_groups_overlapping_bp, '
                        'SUM(min(assembly.stop,bp.stop)-max(assembly.start,bp.start))/1e6 AS overlap_ms ' +
                        overlap_join + ' GROUP BY assembly.process_tag ORDER BY assembly.process_tag LIMIT 10'),
        "overlap_examples": ('SELECT assembly.process_tag,assembly.phase AS assembly_scan,bp.phase AS reconstruction_scan, '
                             'assembly.stream AS assembly_stream,bp.stream AS bp_stream, '
                             '(min(assembly.stop,bp.stop)-max(assembly.start,bp.start))/1e3 AS overlap_us ' +
                             overlap_join + ' ORDER BY assembly.start LIMIT 12'),
        "handoff_apis": ('SELECT ((ranges.globalTid >> 24) << 24) AS process_tag,names.value AS api, '
                         'COUNT(*) AS calls,SUM(runtime."end"-runtime.start)/1e6 AS host_ms ' + runtime_join +
                         f'JOIN StringIds names ON names.id=runtime.nameId WHERE {scope} '
                         "AND (names.value LIKE '%Synchronize%' OR names.value LIKE '%Malloc%' "
                         "OR names.value LIKE '%Free%' OR names.value LIKE '%MemGetInfo%') "
                         'GROUP BY process_tag,names.value ORDER BY process_tag,host_ms DESC LIMIT 40'),
        "correlation": ('SELECT globalPid AS process_tag,COUNT(*) AS total_kernels, '
                        'SUM(EXISTS(SELECT 1 FROM runtime_correlations runtime '
                        'WHERE runtime.correlationId=kernel.correlationId '
                        'AND runtime.process_tag=kernel.globalPid)) AS runtime_correlated '
                        'FROM CUPTI_ACTIVITY_KIND_KERNEL kernel WHERE globalPid IN '
                        '(SELECT DISTINCT process_tag FROM scoped) GROUP BY globalPid LIMIT 10'),
    }
    evidence = dict(database=str(args.database.resolve()), scoped_kernel_query=scoped, queries=queries)
    for name, query in queries.items():
        evidence[name] = [dict(row) for row in connection.execute(query)]
        print(json.dumps({name: evidence[name]}), flush=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    connection.close()


if __name__ == "__main__":
    main()
