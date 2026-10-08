# Performance audit of the tomography pipeline

Notes, probes and run results from the audit of `tomography/` and of tango-ucx underneath it,
started 2026-10-07 on one RTX 2060 SUPER. Nothing here was measured on the target hardware.

| Read | For |
|---|---|
| [NOTES.md](NOTES.md) | The findings in order, with the commands that reproduce them |
| [FOLLOWUP-SUMMARY.md](FOLLOWUP-SUMMARY.md) | The second round: FBP, metadata, layout, scan overlap, transport, each with its own report |
| [THROUGHPUT-TAKEAWAYS.md](THROUGHPUT-TAKEAWAYS.md) | What to do about throughput, ranked |
| [../../tomography/NEXT.md](../../tomography/NEXT.md) | The pipeline planned from these results |

`probes/` holds the scripts. The directories beside the reports hold each run's summaries,
device reports and logs. Detector archives, volumes, reference reconstructions and Nsight
traces are not in the repository (see [.gitignore](.gitignore)), so a report's link to one of
those files leads nowhere here; the report gives the command that produces it.
