# Does input resolution help a DINOv3 detector find small objects?

Part of our group project for 1MD049 Task 1 (DINOv3) on SeaDronesSee.

Most objects in SeaDronesSee (swimmers, buoys, life-saving appliances) are only a few pixels wide in the aerial images. This experiment asks: **if we feed a DINOv3-based detector larger images, does it detect more of them, and what does it cost?**

The detector is LT-DETR from LightlyTrain with a **frozen** DINOv3 backbone. Only the detection head is trained.

## Setup

- **Dataset:** SeaDronesSee object detection, from the Kaggle copy [`ubiratanfilho/sds-dataset`](https://www.kaggle.com/datasets/ubiratanfilho/sds-dataset). Classes: swimmer, boat, jetski, life-saving appliances, buoy.
- **Start point:** one LT-DETR checkpoint trained at 1280 px (frozen backbone).
- **Inference-only resize:** the same 1280 weights evaluated at 1280, 1536 and 1920 px, with no extra training.
- **Fine-tune at size (matched recipe):** from that 1280 checkpoint, 2 more epochs at 1280 (control) and at 1536. Both runs use the same settings (lr 1e-5, 100 warm-up steps, batch 4, class head reused). Only the training size differs.
- **Evaluation:** the same 1446-image validation set (split by source group). Each checkpoint is loaded into a fresh model and evaluated with batch 1, BF16, score threshold 0.0, on one NVIDIA L40S.

## Results

| Model / inference size | AP50 | AP | AP75 | AP_small | Swimmer AP | AR@100 | ms / img |
|---|---:|---:|---:|---:|---:|---:|---:|
| 640 (separate 640 lineage)† | 73.8 | 46.3 | 48.9 | 23.3 | 33.9 | 62.7 | n/a |
| 1280-trained, infer 1280 | 74.9 | 47.6 | 50.2 | 24.8 | 35.8 | 65.2 | 22.0 |
| 1280-trained, infer 1536 (resize only) | 76.2 | 47.3 | 49.9 | 26.8 | 34.2 | 65.0 | 30.0 |
| 1280-trained, infer 1920 (resize only) | 73.0 | 45.1 | 48.1 | 25.3 | 32.5 | 64.1 | 46.4 |
| +2 ep at 1280, infer 1280 (control) | 74.8 | 48.2 | 50.6 | 26.1 | 37.1 | 66.0 | 22.4 |
| **+2 ep at 1536, infer 1536** | **77.3** | **49.3** | **51.8** | **28.0** | **37.3** | **66.0** | 30.0 |

† Different training run, scored with LightlyTrain's own validation during training. It is not a controlled comparison.

Per-class AP50 for the two matched runs (pycocotools on the saved predictions, [`coco_per_class.json`](reports/resolution_comparison/)):

| Class | +2 ep at 1280 | +2 ep at 1536 |
|---|---:|---:|
| Swimmer | 81.3 | 80.8 |
| Boat | 93.8 | 93.8 |
| Jetski | 96.2 | 96.2 |
| Life-saving appliances | 22.9 | **32.4** |
| Buoy | 81.8 | 82.6 |

### What this shows

1. **Fine-tuning at 1536 helps.** Compared with the matched 1280 control, AP50 rises by 2.5 points (74.8 → 77.3) and AP_small by 1.9. Inference becomes about 35% slower (22 → 30 ms per image).
2. **Most of the gain comes from one class.** Life-saving appliances, the smallest and rarest class, gain about 9.5 AP50 points. The other classes barely change.
3. **Resizing at test time alone is not enough.** Running the 1280 weights at 1536 gives part of the AP50 gain (+1.3), but AP and swimmer AP drop. At 1920 every metric is worse, so the head has to be trained at the larger size.
4. A 2048×1152 run was stopped after 1 of 2 epochs (epoch 1: AP50 76.0, compared with 77.0 for 1536 at epoch 1), so it is left out of the table. All rows, including incomplete ones, are in [`resolution_results.csv`](reports/resolution_comparison/resolution_results.csv).

## Files

| File | What it is |
|---|---|
| `train_ltdetr_1536_lr1e5.sbatch` | Slurm job: 2 epochs at 1536 from the 1280 checkpoint. |
| `train_ltdetr_1280_lr1e5_2ep_control.sbatch` | Slurm job: the matched 1280 control. |
| `predict_full_validation_ap50.sbatch` | Slurm job: saves all validation predictions for per-class AP50. |
| `requirements-ltdetr.lock.txt` | Exact package versions of the GPU environment. |
| `scripts/run_ltdetr_frozen.py` | Trains one frozen-backbone LT-DETR stage from earlier weights. |
| `scripts/ltdetr_epoch_validation.py` | Runs validation at real epoch boundaries (LightlyTrain 0.18 only supports a fixed step interval). |
| `scripts/ltdetr_dual_validation.py` | Evaluates each epoch's weights at both 1280 and 1536 without changing the training state. |
| `scripts/ltdetr_early_stopping.py` | Epoch-level mAP early stopping. |
| `scripts/ltdetr_small_object_crop.py` | Object-centred training crops (used only in a 640 variant). |
| `scripts/validate_ltdetr_resolutions.py` | Evaluates one fixed checkpoint at several input sizes. |
| `scripts/predict_full_validation.py` | Saves every validation prediction in COCO results format. |
| `scripts/coco_per_class_ap.py` | Computes overall and per-class AP and AP50 with pycocotools on the CPU. |
| `scripts/make_resolution_summary.py` | Collects all results into the CSV and the slide table. |
| `scripts/ltdetr_validation_email.py` | Writes comparison tables and sends Slurm summary emails (no credentials). |
| `reports/resolution_comparison/` | Results: `resolution_results.csv`, `slide_table.md` and per-class AP JSON files. |

## Running

The jobs were run on UPPMAX (account `uppmax2026-1-299`, L40S GPUs) from a separate working copy. The paths in the `.sbatch` files (`scripts/`, `runs/`, `.venv-gpu/`) are relative to that copy, and the training checkpoints in `runs/` are too large for this repository. To run a job again, create a venv from `requirements-ltdetr.lock.txt` and submit it from a directory containing `scripts/` and the 1280 checkpoint.
