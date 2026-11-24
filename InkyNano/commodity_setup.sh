#!/usr/bin/env bash
set -euo pipefail
ROOT="$(realpath "${1:?workspace required}")"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT"
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python -m venv --system-site-packages "$ROOT/venv"
PY="$ROOT/venv/bin/python"
"$PY" -c 'import torch; assert torch.cuda.device_count()==1; assert torch.cuda.get_device_properties(0).total_memory>=75*1024**3; print(torch.__version__,torch.cuda.get_device_name(0))'
REV=d8f10a82213a3a9db48385aba3a4b57ab535de50
if [[ ! -d "$ROOT/transformers/.git" ]]; then
  git init "$ROOT/transformers"
  git -C "$ROOT/transformers" remote add origin https://github.com/huggingface/transformers.git
fi
git -C "$ROOT/transformers" fetch --depth 1 origin "$REV"
git -C "$ROOT/transformers" checkout --detach FETCH_HEAD
test "$(git -C "$ROOT/transformers" rev-parse HEAD)" = "$REV"
"$PY" -m pip install --no-deps "$ROOT/transformers"
"$PY" -m pip install huggingface-hub==1.31.0 tokenizers==0.23.1 safetensors==0.8.0 accelerate numpy packaging pyyaml regex requests tqdm
"$PY" -c 'import transformers; from transformers import Inkling2LoopForConditionalGeneration; from transformers.core_model_loading import Inkling2LoopDequantizeExperts; assert transformers.__version__=="5.24.0.dev0"'
printf '{"commit":"%s"}\n' "$REV" > "$ROOT/runtime_receipt.json"
"$PY" - "$ROOT" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(sys.argv[1])
snapshot_download("thinkingmachines/Inkling-2-Mini",revision="ed40e7da625f594fc9c4990f5f66be24873c22d7",
 local_dir=root/"source_snapshot",allow_patterns=["config.json","hf_quant_config.json","model.safetensors.index.json","model-*.safetensors","tokenizer*","special_tokens_map.json","generation_config.json","chat_template*","preprocessor_config.json"],max_workers=4,token=False)
assert (root/"source_snapshot"/"model-00001-of-00001.safetensors").stat().st_size==19526019168
PY
"$PY" "$HERE/inventory.py" --local "$ROOT/source_snapshot" --output "$ROOT/inventory.json"
"$PY" -m unittest discover -s "$HERE" -p 'test_*.py' -v
"$PY" -m pip freeze > "$ROOT/runtime_packages.txt"
