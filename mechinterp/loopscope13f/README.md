# LGA-10F: curvature level-set donor repair

Source-only FP32 engineering study on one untrained model, with native-state
FP64 geometry and mixed-precision donor/sham construction.

- [PROTOCOL.md](PROTOCOL.md): prospective protocol (pre-source)
- [manifold.py](manifold.py): fixed-budget curvature level-set donor solver
- [source61069.py](source61069.py): three-arm original native-FP32 endpoint
- [audit61069.py](audit61069.py): independent NumPy reconstruction
- [IMPLEMENTATION_FREEZE.md](IMPLEMENTATION_FREEZE.md): one-shot source guard

Run model tests in **GitHub Actions**, not on local GPU. The preflight
PR runs the tests on exposed seed 61064 and skips source generation.
Only the first source-branch implementation push initializes engineering
seed 61069 and archives the complete first-run source. No trained
model root or protected science seed is touched.
