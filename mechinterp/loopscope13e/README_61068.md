# LGA-10E 61068: fresh excluded engineering source

- [Frozen prospective protocol](PROTOCOL_61068.md)
- [Registered implementation freeze](IMPLEMENTATION_FREEZE_61068.md)
- [Native-FP32 endpoint source runner](source61068.py)
- [Independent NumPy auditor](audit61068.py)

Staging CI tests the code without generating the excluded source. The first
frozen implementation push is the only workflow entry that opens seed 61068.
Subsequent PRs and reruns run tests alone.

The code is CPU-only and expects Python 3.11, Torch 2.7.1+cpu and NumPy 2.2.6.
The preserved source artifact includes all failures; threshold shortfalls do
not fail a correctly audited Actions job. Protected science seeds stay closed.

The primary arm is **mixed-precision construction** with fully native-FP32
continuation. The fixed-donor comparator uses FP32 selection filtering atop
FP64 source geometry, and neither selection arm claims FP32-only preparation.
