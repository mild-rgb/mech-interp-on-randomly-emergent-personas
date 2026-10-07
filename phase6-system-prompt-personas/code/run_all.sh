#!/bin/bash
# phase 6 — the whole VM pipeline, run in the background on a Colab A100 80GB:
#   inputs from HF -> generation (vLLM, Qwen3-8B) -> capture (HF transformers) -> judging (vLLM, Gemma 4 31B) -> upload.
# Every stage skips work already on disk, so the script can be re-run after an interruption.
# Needs /content/hf_token. Inputs live in the private dataset repo $REPO under inputs/.
set -euo pipefail
REPO=mild-rgb/phase6-system-prompt-personas-qwen3-8b; P3=mild-rgb/phase3-layer0-persona-direction-qwen3-8b
export WORK=/content/p6 HF_TOKEN=$(cat /content/hf_token) VLLM_ENABLE_V1_MULTIPROCESSING=0 TOKENIZERS_PARALLELISM=false
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
mkdir -p $WORK/p3/features
stamp() { echo "=== [$(date +%H:%M:%S)] $*"; }
stamp install
python3 -c "import vllm" 2>/dev/null || pip install -q vllm==0.29.0
python3 -c "import torch, transformers, vllm; print('torch', torch.__version__, '| transformers', transformers.__version__, '| vllm', vllm.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv
stamp inputs
python3 - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download
W = os.environ["WORK"]
for f in ["requests.jsonl", "steered_match.jsonl", "gate_rows.json", "steered_capture_rows.json", "rubric.txt", "probe_l0_direction.json",
          "job_gen_vllm.py", "job_capture_hf.py", "job_judge_vllm.py"]:
    shutil.copy(hf_hub_download("mild-rgb/phase6-system-prompt-personas-qwen3-8b", f"inputs/{f}", repo_type="dataset"), f"{W}/{f}")
for f in ["gen.jsonl", "features/X_000.npy"]:
    shutil.copy(hf_hub_download("mild-rgb/phase3-layer0-persona-direction-qwen3-8b", f, repo_type="dataset"), f"{W}/p3/{f}")
print("inputs ok", sorted(os.listdir(W)))
EOF
stamp generation
N=$(wc -l < $WORK/requests.jsonl)
if [ ! -f $WORK/gen.jsonl ] || [ "$(wc -l < $WORK/gen.jsonl)" -lt "$N" ]; then python3 -u $WORK/job_gen_vllm.py; fi
stamp capture
if [ ! -f $WORK/features_steered/features_meta.json ]; then python3 -u $WORK/job_capture_hf.py; fi
stamp upload-features
python3 - <<'EOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); W = os.environ["WORK"]
api.upload_file(path_or_fileobj=f"{W}/gen.jsonl", path_in_repo="gen.jsonl", repo_id="mild-rgb/phase6-system-prompt-personas-qwen3-8b", repo_type="dataset")
for d in ("features", "features_steered"):
    api.upload_folder(folder_path=f"{W}/{d}", path_in_repo=d, repo_id="mild-rgb/phase6-system-prompt-personas-qwen3-8b", repo_type="dataset")
print("features uploaded")
EOF
stamp judge
if [ ! -f $WORK/judge/judge_match.json ]; then python3 -u $WORK/job_judge_vllm.py; fi
stamp upload-judge
python3 - <<'EOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); W = os.environ["WORK"]
api.upload_folder(folder_path=f"{W}/judge", path_in_repo="judge", repo_id="mild-rgb/phase6-system-prompt-personas-qwen3-8b", repo_type="dataset")
for f in ["gen_meta.json"]:
    if os.path.exists(f"{W}/{f}"): api.upload_file(path_or_fileobj=f"{W}/{f}", path_in_repo=f, repo_id="mild-rgb/phase6-system-prompt-personas-qwen3-8b", repo_type="dataset")
print("judge uploaded")
EOF
stamp ALL DONE
