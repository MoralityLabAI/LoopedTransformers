# NAN-COMP-04: staged late-evidence reciprocal gym

**Status:** source-only implementation and protocol; no pretrained model inference, BPD intervention, or sealed-test read. Original NAN-COMP-02/03 evidence remains unchanged. The source-locked model runner requires a separate explicit --execute-development flag before loading weights.

The mathematical source family is f(x)=1+a*x^m+c*x^(m+2), with odd m=3,5 and integer nonzero a,c. N(f)=f(-x) and R(f)=1/f agree through degree 2m-1, then differ at degree 2m by a² and at degree 2m+2 by 2ac. The separate numeric question asks for that last coefficient in the reciprocal of an independently chosen third query input. The source-only oracle independently verifies truncated polynomial convolution. Classification and numerical generation use full-vocabulary greedy decoding, with strict anchored grammars.

## Source-only setup

From the LoopedTransformers repository root, using Python 3.12:

~~~shell
python -m unittest discover -s experiments/nanbeige/nan_comp04/tests -p 'test_*.py' -v
python -m experiments.nanbeige.nan_comp04.cohort --out runs/nanbeige/nan_comp04/source
python -m experiments.nanbeige.nan_comp04.run --source-dir runs/nanbeige/nan_comp04/source --runs runs/nanbeige/nan_comp04/native --compactification-root LOCAL_ARG02_CHECKOUT --snapshot LOCAL_PINNED_NANBEIGE_SNAPSHOT --backend bf16
~~~

The last command is a dry-run unless --execute-development is explicitly supplied. It validates complete source identities and prints the selected two DEV roots without importing CUDA or loading model weights.

The generator creates 4 CAL roots and 16 DEV roots, yielding 20 x 11 = 220 model-visible source tasks. Host-only program and coefficient labels live in a separate JSONL. Forty future confirmation root IDs are listed in config.json but are never generated or scored at this stage.

## Native RTX 3050 pilot, after source-lock review

Use the previously tested pinned Nanbeige4.2-3B snapshot (revision 451ed48c3273ecef7ea8faaa43c31ce529763bb1), the matching ARG-02 checkout and a working Transformers 4.46.3 / Torch CUDA / bitsandbytes environment. Supply actual local paths.

~~~shell
python -m experiments.nanbeige.nan_comp04.run --source-dir runs/nanbeige/nan_comp04/source --runs runs/nanbeige/nan_comp04/native --compactification-root LOCAL_ARG02_CHECKOUT --snapshot LOCAL_PINNED_NANBEIGE_SNAPSHOT --backend bf16 --split development --limit-roots 2 --execute-development
python -m experiments.nanbeige.nan_comp04.run --source-dir runs/nanbeige/nan_comp04/source --runs runs/nanbeige/nan_comp04/native --compactification-root LOCAL_ARG02_CHECKOUT --snapshot LOCAL_PINNED_NANBEIGE_SNAPSHOT --backend nf4 --split development --limit-roots 2 --execute-development
python -m experiments.nanbeige.nan_comp04.grade --source-dir runs/nanbeige/nan_comp04/source --runs runs/nanbeige/nan_comp04/native --backend bf16 --split development --out runs/nanbeige/nan_comp04/grade_bf16_pilot.json
python -m experiments.nanbeige.nan_comp04.grade --source-dir runs/nanbeige/nan_comp04/source --runs runs/nanbeige/nan_comp04/native --backend nf4 --split development --out runs/nanbeige/nan_comp04/grade_nf4_pilot.json
~~~

Start with the two-root pilot. Use a new output directory with --limit-roots 0 for complete 16-root DEV inference because the run manifest pins the chosen root set. All model-side outputs are source-hashed, and the CPU host grader runs separately.

The model runner imports only public source data, reuses the CPU-resident full-vocabulary LM head and native two-pass shared-weight decoder, deduplicates identical early N/R and output-erased prompts, enforces tokenizer-prefix and length parity, and refuses to run without a committed SOURCE_LOCK.json. The numeric parser accepts an anchored square-bracketed signed integer, allowing an optional balanced bold wrapper. Formatting failure is recorded as an incorrect outcome.

## Evidence policy

The primary development prerequisites are N/R both-target correctness at the first distinguishing coefficient, exact numeric query coefficient correctness and reciprocal competence. The early-evidence N/R pair and output-erased pair have identical visible prompts, so their same-root joint correctness is impossible. The original exposed-data R score-bias adjustment is a post-hoc diagnostic and cannot be selected on DEV. Any fixed readout prior must be fitted on CAL and frozen before DEV.

Use source roots as the unit in EvalMath FSI-01. Twenty total roots, including only sixteen DEV roots, support pilot-level conclusions. Independent sham and fresh confirmation are needed before mechanistic claims. The 168-case ARG-03 held-out test stays sealed. BPD_transfer_enabled remains false until a separately registered experiment passes the competence gate.

## Prior evidence

The umbrella folder at ../ contains the original research note and an exact-blob archive of nine NAN-COMP-03 source and result files, plus PROVENANCE.json. The complete per-case BF16/NF4 pretrained receipts remain at ../../nan_comp03/receipts/. They are linked rather than rewritten or regraded here.
