#!/usr/bin/env bash
# Inference only (threshold 0, full predictions) for per-class AP50: 1280 +2ep @1280 and 1536 +2ep @1536.
#SBATCH --account=uppmax2026-1-299
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=ltdetr_ap50_pred
#SBATCH --output=/gorilla/proj/uppmax2026-1-299/hsuanchu/dinov3_outputs/ltdetr_ap50_pred_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4
module load Python/3.12.3-GCCcore-13.3.0
OUT=/gorilla/proj/uppmax2026-1-299/hsuanchu/dinov3_outputs/per_class_ap50_${SLURM_JOB_ID}
.venv-gpu/bin/python -u scripts/predict_full_validation.py --size 1280 \
  --source runs/ltdetr_seadronessee/full_1280_frozen_lr1e5_2ep_control/dual_validation/epoch_002_step_461/results/source_best.pt \
  --out $OUT/train1280_e2_infer1280
.venv-gpu/bin/python -u scripts/predict_full_validation.py --size 1536 \
  --source runs/ltdetr_seadronessee/full_1536_frozen_lr1e5/dual_validation/epoch_002_step_461/results/source_best.pt \
  --out $OUT/train1536_e2_infer1536
echo "Inference done; exiting to release GPU."
