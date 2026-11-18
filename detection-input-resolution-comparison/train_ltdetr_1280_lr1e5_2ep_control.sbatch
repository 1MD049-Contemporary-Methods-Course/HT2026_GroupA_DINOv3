#!/usr/bin/env bash
# Control for the 1536 fine-tune: identical recipe, train size 1280.
#SBATCH --account=uppmax2026-1-299
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=ltdetr_1280_ctrl
#SBATCH --output=runs/ltdetr_1280_ctrl_%j.log
#SBATCH --mail-type=END,FAIL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=2
export LTDETR_EVAL_PRECISION=bf16-mixed
module load Python/3.12.3-GCCcore-13.3.0
nvidia-smi
.venv-gpu/bin/python -u scripts/run_ltdetr_frozen.py \
  --size 1280 --batch 4 --workers 4 --precision bf16-mixed \
  --lr 1e-5 --warmup-steps 100 --reuse-class-head --dual-validation --max-epochs 2 \
  --checkpoint runs/ltdetr_seadronessee/full_1280_frozen/checkpoints/best.ckpt \
  --out runs/ltdetr_seadronessee/full_1280_frozen_lr1e5_2ep_control
