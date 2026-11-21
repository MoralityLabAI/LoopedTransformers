# COMP14 CI checkout bytes

The first Linux CI byte check at freeze commit9bcfec1 failed on the reused
arg03_nanbeige.py dependency; its five source/parser/grader tests passed.
Both original gym workflows passed too. The frozen Windows worker's exact
20-file/13-dependency byte check passed before launch and remains unchanged.

The two legacy dark_fiber_gyms files (arg03_nanbeige.py and q4_3050.py) have
LF Git blobs but CRLF Windows working-tree bytes. The frozen hashes bind those
actual Windows runtime bytes. This independent CI-only checker accepts exact
bytes for all34 critical/host/dependency bindings, or one explicit whole-file
LF-to-CRLF conversion for those two dependencies only. All other bytes and
paths remain exact. Contract tests reject semantic changes, conversions on
unlisted paths and conversions of new critical sources.

No frozen source, manifest, prompt, parser, case, readout, gate or live worker
is edited. No model call is added or replayed. Failed CI runs44858604523 and
44858608751 remain in the record. This is checkout validation, not a change
to inference or scientific analysis.
