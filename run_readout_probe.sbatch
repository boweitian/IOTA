#!/bin/bash
#SBATCH --job-name=iota-readout
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=angliece
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --output=/beacon-projects/worldmodel/btian/2.3_Monad/IOTA/output/readout-probe-slurm-%j.log
set -euo pipefail
cd /beacon-projects/worldmodel/btian/2.3_Monad/IOTA
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
/beacon-projects/worldmodel/btian/miniforge3/envs/pathorob/bin/python probe_readout.py --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --cache-dir /beacon-projects/worldmodel/btian/hf_cache/hub --output output/readout_probe_v2
