#!/usr/bin/env bash
#SBATCH --account=uppmax2026-1-299
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=ltdetr_1536_lr1e5
#SBATCH --output=runs/ltdetr_1536_lr1e5_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=2
module load Python/3.12.3-GCCcore-13.3.0
.venv-gpu/bin/python -u scripts/run_ltdetr_frozen.py \
  --size 1536 --batch 4 --workers 4 --lr 1e-5 --warmup-steps 100 \
  --reuse-class-head --dual-validation --max-epochs 2 \
  --checkpoint runs/ltdetr_seadronessee/full_1280_frozen/checkpoints/best.ckpt \
  --out runs/ltdetr_seadronessee/full_1536_frozen_lr1e5
