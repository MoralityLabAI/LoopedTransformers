# Pre-inference Windows owned-launcher admission amendment

The old host check excluded the actual calling PID but counted its Windows
virtual-environment launcher as a separate live host. A stdlib-only startup
fixture using the real GPU venv reproduced this before any CACHE32 model ran.
The fixture always exits in sitecustomize, before the host module executes.

The corrected inventory exempts that parent only for the exact registered
host entry, current-venv executable, identical argument suffix, actual parent
relationship and a creation time older than the actual host. Kernel handles
bind creation times. Missing/ambiguous metadata gives no parent exemption.
Foreign workers/servers/hosts/audits, including the real active GPU worker in
the fixture, remain blocked. No blanket parent/process-name exemption.

Only ensure_inventory_clear's delegate changes in predecessors.py; normalized
AST comparison proves all predecessor wait code exact. Every other original
critical source, numeric method, full1,147-job plan, scope, assets and criterion
is byte-exact. Freeze bookkeeping validates historical ENGINEERING.json
against preserved original sources, then binds this separate amendment and
its new proof. The historical17-capture numerical CPU preflight is retained
as historical evidence, not relabelled as locally rerun.

Nine existing cold/scope contracts and eight inventory/AST contracts passed
under a128-MiB private cap,5-GiB admission,4-GiB reserve and one Below Normal
CPU thread. The actual Windows startup fixture used the same limits. Neither
imported Torch/NumPy or executed a model/host evaluation.

Only the authenticated unstarted CACHE32 waiter was retired, after proving
both worker permits/runtimes absent and its artifact root empty. Original
source seal4993a010890cc4a6d5b257bcdea61cb63fd69282603f423a1f9c2762589aeddd
and its unstarted runtime are archived. Original NF4/model/supervisor, BF31
revision2 and the math process are untouched. A fresh cached source seal must
be pushed before its replacement waiter can pass publication admission.
All previous failures remain; no response rescue or new neural call.
The original168-case test stays sealed and the continuous goal remains active.
