#!/bin/bash
#SBATCH --job-name=iota-mvp
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/slurm-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA
export HF_HOME=/beacon-projects/worldmodel/btian/hf_cache
export PYTHONUNBUFFERED=1
nvidia-smi -L
PY=/beacon-projects/worldmodel/btian/miniforge3/envs/pathorob/bin/python
"$PY" -c 'import torch, transformers; print(torch.__version__, transformers.__version__); assert torch.cuda.is_available()'
"$PY" demo.py
"$PY" mvp.py --model Qwen/Qwen2.5-3B --cache_dir "$HF_HOME/hub" --output_dir ./output --dtype bf16
