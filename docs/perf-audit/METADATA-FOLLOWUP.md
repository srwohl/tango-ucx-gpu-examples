# Python metadata follow-up

Measured 2026-10-07 using Python 3.14.7, the current `tango-ucx` GPU environment,
and real retained host UCX batches. This change amortizes Python records-view
construction within a batch. It does not change the native tomography compute
stages, their binary metadata processing, publisher sends, or UCX discovery.

## Change and ownership

`tango-ucx/python/tango_ucx/subscription.py:245` now lazily retains one private
native records array per public `Batch`. Each property access returns a fresh
read-only NumPy `.view()` sharing that storage, with an independent recursive
dtype clone from `dtype.newbyteorder("|")`. The `"|"` mode leaves byte order
unchanged. Changing a returned array's shape, assigning a different dtype, or
renaming its top-level/nested/subarray dtype fields leaves subsequent property
descriptors intact. Schema tests also check mixed endianness, alignment, padding
and field titles. `np.dtype(existing, copy=True)` returns the same dtype in this
NumPy version; Python `deepcopy` costs about 30 microseconds here, whereas the
native recursive clone costs below a microsecond in the isolated schema test.
The first access still invokes the existing native dtype construction and its
alignment, reserved-name, object-field and size checks. Failed construction is
not cached. A copied batch starts with its own uncached records storage.

The private cached array's base is the native batch; it does not retain the public
Python wrapper. A real fixture test confirms the public wrapper is collected
while exported records keep receive slots held, and records remain usable after
subscription close. The cache stays in Python because native `PythonBatch` can
be retained by DLPack consumers and released on a foreign thread; adding Python
object members there would require additional GIL/finalization handling.

## Measurements and limits

Both runs use the same native extension built from source revision
`e3133925f1ec4dc1d4e3b3f5c72b10dbff96571f`, SHA256
`9c91d67d71e1ac63dc78be2d65d10aff62f237843a9b6cbb7d8d6aab0a874d93`.
The baseline is an isolated package containing the unchanged `HEAD` Python
wrapper; the after run uses `build-gpu/python`. The examples' staged installation
is untouched. The package locations are recorded in the raw JSON artifacts.

Medians of three samples, 1000 operations per sample; all values in microseconds
per frame except the getter-only row, which is microseconds per property access:

| Pattern | Frames | Before | After |
| --- | ---: | ---: | ---: |
| Repeated public records getter only | 1 | 31.65 | 1.276 |
| Repeated public records getter only | 16 | 38.84 | 1.793 |
| Repeated public records getter only | 128 | 37.18 | 1.487 |
| Fresh batch wrapper, records property for each frame | 16 | 35.80 | 4.91 |
| Fresh batch wrapper, records property for each frame | 128 | 35.73 | 2.23 |
| Fresh batch wrapper, records accessed once then iterated | 16 | 2.99 | 3.39 |
| Fresh batch wrapper, records accessed once then iterated | 128 | 0.844 | 0.850 |
| Previously cached records iterated directly | 128 | 0.552 | 0.587 |
| Native `Batch.index` through wrapper's private native handle | 128 | 0.227 | 0.186 |

Fresh-wrapper measurements construct a new public wrapper around the same
retained native batch for every operation. This includes lazy dtype/view
construction without timing transport or publisher work. At 128 frames the
repeated-per-frame pattern improves about 16 times even including first access.
Reading records exactly once per fresh batch has no established speedup: its
first construction remains. Small differences in the once-per-batch rows should
not be treated as a transport or reconstruction effect.

The direct native `records` getter remains about 30–40 microseconds. The cached
public getter measures only 1.28–1.79 microseconds here. These are isolated Python
metadata timings; they do not establish an end-to-end tomography improvement.
CPU activity was not isolated, and the after probe overlapped a regression suite;
use the raw sample ranges when assessing small differences. The large repeated
getter gain is distinct from those small variations.

The publisher fixture uses a seed field, not tomography's application schema.
All measured batches remain read-only and share native record storage, and
terminal health reports no failure or quarantined bytes. Connection times are
excluded from getter timing; initial lazy imports remain about 1.23 seconds.

## Validation

The targeted real-fixture tests check storage sharing and copied storage
independence; read-only enforcement; wrapper collection, receive-slot retention
and lifetime after close; independent shape/dtype assignment and nested dtype
names; mixed-endian/title/padding/alignment preservation; and alignment
rejection on two attempted first accesses. Existing GPU tests exercise records
and DLPack lifetime interactions without adding Python state to native batches.

The final cache passed 40 host subscription/opaque/nodb regressions, followed by
the additional title/padding/alignment schema-isolation test. Independent final
checks also passed all seven GPU tests and the three cached-records fixture tests.
Shape/dtype mutation assertions intentionally exercise
APIs deprecated by NumPy 2.5 and emit two deprecation warnings on this build.
Repository architecture checks and `git diff --check` pass.

```bash
cd tango-ucx
pixi run -e gpu cmake --build build-gpu --target _core fixture_device
PYTHONPATH=build-gpu/python \
  TANGO_UCX_FIXTURE_DEVICE=build-gpu/tests/fixture_device \
  pixi run -e gpu pytest python/tests/test_opaque.py python/tests/test_nodb.py \
    python/tests/test_gpu.py python/tests/test_subscription.py -q

pixi run -e gpu python ../tango-ucx-gpu-examples/docs/perf-audit/probes/metadata_access.py \
  --python-package build-gpu/python --iterations 1000 \
  --output ../tango-ucx-gpu-examples/docs/perf-audit/metadata-current-source-after.json
```

The probe's explicit `--python-package` overrides the examples network helper's
staged-package precedence. Setting only `PYTHONPATH` previously loaded the staged
extension anyway. Its native-index comparison now checks `batch._native.index`;
the public Python wrapper does not expose that method.

Raw measurements: `metadata-current-source-before.json` and
`metadata-current-source-after.json`. A baseline package can be recreated by
copying `build-gpu/python/tango_ucx` to a temporary directory and restoring that
copy's `subscription.py` from the source revision above; pass its containing
directory with `--python-package`. Do not run CMake install into examples/stage
for this comparison.
