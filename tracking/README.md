# Frozen DINOv3 features for keeping IDs on SeaDronesSee

This is my part of our group project for 1MD049 Task 1 (DINOv3). In it I test one question: **can frozen DINOv3 appearance features help a simple tracker keep the same ID on the same target across frames?**

I give the tracker the ground truth boxes from the dataset, so I only test the matching step. DINOv3 does one job: it turns each box crop into a 384 dimensional appearance vector. I wrote the motion model, the gate, the matching and the track management myself. I do not train anything.

I compare two versions of the tracker on the same boxes:

- **Geometry only**: matches boxes by predicted position and box overlap.
- **Geometry + DINOv3**: the same, plus a cost for how different two crops look.

## Files


| File | What it is |
|---|---|
| `README.md` | This file. |
| `Tracking.ipynb` | My whole pipeline, written for Kaggle, saved with the outputs of my run. Cells 0 to 18 run the main experiment and cells 19 to 22 are my follow up on `thu0109`. |
| `identity_labels_template.csv` | My manual identity labels for `thu0109`: 150 boxes, `gt_id` 1 to 3. I filled in the template from the result folder and kept its file name, so it matches the path in `GT_ID_FILES`. |
| `thu0109_automatic_comparison.zip` | Output of the follow up (cells 19 to 22): the scores for all eight settings, the check of every link DINOv3 changed and the errors left in each method. |

### Notebook layout

I count cells from 0, including markdown cells.

| Cell | Heading in the notebook | What it does |
|---|---|---|
| 0 | SeaDronesSee × DINOv3: tracking on two new clips | Overview and Kaggle steps |
| 2 | 1. Plan and settings | Settings and the six tracker variants |
| 3 | 2. Tracker and data tools (this heading sits in cell 4, under the code) | `AssociationTracker`, frame selection, feature cache, `trajectory_links`, `score_identities` |
| 6 | 3. Load the clips and check the data | Finds the dataset, checks hashes, builds the two clips, saves the plan |
| 8 | 4. Load cached features | Loads features from the output folder or an attached result ZIP |
| 10 | 5. Extract the 224 CLS features once on the GPU | Crops the boxes and runs DINOv3 (skipped when everything is cached) |
| 12 | 6. Run the six settings and three constant feature checks | Runs the trackers and the checks |
| 14 | 7. Score with my manual identity labels | Link precision, recall and F1 for labelled clips |
| 16 | 8. Six panel video and labelling frames | Comparison video and numbered frames for labelling |
| 18 | 9. Save the results | Saves the config and zips everything |
| 19 to 21 | no heading | Follow up on `thu0109`: motion model and gate |
| 22 | 10. Score the eight settings with my manual labels (code comment) | Scores the follow up settings and checks every link DINOv3 changed |

## Data

- **Source:** the Kaggle dataset [`ubiratanfilho/sds-dataset`](https://www.kaggle.com/datasets/ubiratanfilho/sds-dataset), a copy of the SeaDronesSee object detection set. The notebook finds the folder with `images/` and `annotations/` and checks the SHA-256 of `instances_train.json` and `instances_val.json` against `EXPECTED_ANNOTATION_HASHES`, so every run uses the same annotations.
- **Building the clips:** the detection set is made of single images, but each image record says which video it came from (`source.video`) and its frame number (`source.frame_no`). `read_groups()` groups the train and val images by video and sorts them by frame number, and `select_context()` takes the frames I ask for.
- **My two clips** (I picked them from the annotation metadata before I ran any tracking):

  | Name | Source video | Source frames | Frames | Boxes | Targets |
  |---|---|---|---|---|---|
  | `sat0131` | `NorthSea_Sat_DJI_0131.MP4` | 0 to 735, every 15th | 50 | 390 (84 small, 156 medium, 150 large) | not labelled |
  | `thu0109` | `NorthSea_Thu_DJI_0109.MP4` | 2145 to 2880, every 15th | 50 | 150 (all medium) | 3, labelled by me |

  All 100 frames are 3840×2160 images from the `train` split.
- **Frame selection:** every frame I ask for has to exist exactly once, otherwise the notebook stops. I clip the boxes to the image and log boxes that become empty in `skipped_boxes.csv` (there were none). I drop annotations with `category_id == 0` or `iscrowd`.
- **Identities:** the detection set has boxes but no IDs across frames, so I labelled the identities myself (see [Labelling](#labelling)).

## Method

### 1. Appearance features

- Model: `facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-S/16) from Hugging Face `transformers`. I pin it to one commit and use it for both clips; my run used `114c1379950215c8b35dfcd4e90a5c251dde0d32`.
- Crop: I cut each box out of the original image and pad it to 224×224 with bicubic resizing, keeping its shape. The padding colour is RGB (124, 116, 104).
- I call the processor with `do_resize=False` and `do_center_crop=False`, so the crop is not resized a second time, and I check that the model input really is 224×224.
- Feature: `pooler_output` (the CLS token), L2 normalised, 384 dimensions, in float32 without autocast or TF32.
- The weights are frozen (`requires_grad_(False)` and `torch.inference_mode()`).
- I extract the features once per clip and all settings share them. They are saved after every batch, so I can resume if a session stops.

### 2. Tracker (`AssociationTracker`)

I divide the boxes by the image width and height, so distances are fractions of the image size. Time is the source frame number. For each new frame:

1. **Predict.** I move each track's centre along its velocity: `state + velocity × Δt × prediction_factor`. The box size stays the same. The velocity is a running average, `0.5 × old + 0.5 × newest`. With `prediction_decay < 1`, I damp the velocity for tracks that missed frames.
2. **Gate.** A box can only match a track if its centre is within `MAX_CENTER_DISTANCE` (**0.12**) of the predicted centre.
3. **Cost.**
   - Geometry: `0.65 * (1 - IoU) + 0.35 * min(distance / gate, 1)`
   - Appearance (DINOv3 versions): `(1 - cosine_similarity) / 2` against the track's stored feature. After each match I update the stored feature to `normalise(0.8 × old + 0.2 × new)`.
   - Total: `geometry + APPEARANCE_WEIGHT × appearance`, with `APPEARANCE_WEIGHT = 0.55`.
4. **Match.** `gated_assignment()` does one to one matching with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). It first makes as many matches inside the gate as possible and then picks the cheapest of those.
5. **IDs.** A matched box keeps its track's ID. A box with no match starts a new track with a new ID. I delete a track after it misses more than `max_missed` frames.

The tracker does not use the class labels.

### 3. Settings

**Main experiment (cell 12), both clips, gate 0.12.** I run each track management setting with and without DINOv3:

| Track management | Geometry only | With DINOv3 |
|---|---|---|
| Keep a lost track for 2 frames, constant velocity | `geometry_keep2` | `dino224_keep2` |
| Keep it for 4 frames, constant velocity | `geometry_keep4` | `dino224_keep4` |
| Keep it for 4 frames, velocity damped by 0.5 per missed frame | `geometry_damped_keep4` | `dino224_damped_keep4` |

**Follow up (cells 19 to 22), only `thu0109`, `max_missed = 2`.** Here I change the motion model and the gate:

| Setting | Motion model | Gate |
|---|---|---|
| `cv_gate12` | constant velocity | 0.12 |
| `lastpos_gate12` | last seen position (velocity set to 0) | 0.12 |
| `cv_gate16` | constant velocity | 0.16 |
| `lastpos_gate16` | last seen position | 0.16 |

Each setting has a `_geometry` and a `_dino` version. Before the new settings, cell 20 checks that I get the original `geometry_keep2` and `dino224_keep2` results back exactly.

### 4. Constant feature check

For every setting I also run the DINOv3 tracker with the same feature for every box. Then appearance cannot prefer any box, so the result has to be identical to geometry only. All 6 + 4 checks passed.

## Evaluation

A **link** is two boxes in a row that share an ID. For example, if ID 5 is on box A and next shows up on box B, then A→B is one link. A link can skip frames, and I count those gap links separately.

- I get the true links from my `gt_id` labels. `thu0109` has 3 targets × 49 steps = **147 true links**.
- I get the predicted links from the tracker IDs.
- `correct_links` are links that are both predicted and true, `wrong_links` are predicted links that are not true, and `missing_true_links` are true links the tracker did not make.
- From these I compute precision, recall and F1 in `score_identities()`.

## How to run (Kaggle)

1. Import `Tracking.ipynb` into Kaggle.
2. **Add Input**: `ubiratanfilho/sds-dataset`.
3. Set **Accelerator** to GPU T4, turn **Internet** on, and turn on the `HF_TOKEN` secret under **Add-ons → Secrets**. The Hugging Face account needs access to the DINOv3 model.
4. **Run All**.
5. Download `SeaDronesSee_MOT_NewVideos_Results_<key>.zip` from `/kaggle/working`.

**Running again without a GPU.** I keep the dataset attached and also add `SeaDronesSee_MOT_NewVideos_Results_eaafa4cd8dfb3eda3339.zip` as an Input, without unzipping it. Cell 8 loads the features from `feature_cache/`. It only uses a cache if the images, boxes, model settings and box order all match, and it never mixes model commits. When everything is cached, the extraction cell skips the model, the token and the GPU.

**Scoring.** I upload `identity_labels_template.csv` as a Kaggle dataset and set its path in cell 2. In my run it was:

```python
GT_ID_FILES = {'sat0131': None, 'thu0109': '/kaggle/input/datasets/dzhang01jan/manual-label/identity_labels_template.csv'}
```

If you upload it under another name, change this path.

Cell 14 then writes `identity_link_metrics.csv`. Clips set to `None` are skipped.

**Follow up (cells 19 to 22).** These need cell 2 (settings and imports) and cell 3 (tracker code). If the main run is still in memory, cell 19 uses it. After a restart, I set `RESULT_ZIP` to the result ZIP and it loads from there. Cells 20 and 21 run the eight settings and count IDs and changed links. Cell 22 reads the label file from `GT_ID_FILES['thu0109']` and:

- scores all eight settings;
- checks each link DINOv3 added or removed;
- lists the wrong and missing links left in each method;
- updates `experiment_protocol.json` and zips `thu0109_automatic_comparison.zip` again.

### Main settings (cell 2)

| Variable | Value | What it does |
|---|---|---|
| `SEQUENCE_PLAN` | the two clips above | Video, start frame and number of frames (2 to 300). `frame_step` stays 15. |
| `DATA_ROOT` | `None` | Folder of the dataset, if it is not found under `/kaggle/input`. |
| `MODEL_ID` / `MODEL_REVISION` | ViT-S/16 / `main` | Pinned to one commit on the first run. |
| `INPUT_SIZE` | 224 | Crop size. |
| `FEATURE_BATCH_SIZE` | 16 | Crops per extraction batch. I use 8 if the GPU runs out of memory. |
| `APPEARANCE_WEIGHT` | 0.55 | Weight of the DINOv3 cost in the `dino224_*` versions. |
| `MAX_CENTER_DISTANCE` | 0.12 | Gate, as a fraction of the image size. |
| `GT_ID_FILES` | `thu0109`: my label file, `sat0131`: `None` | Paths to the filled label files. |
| `PLAYBACK_FPS` | 5 | Speed of the comparison video. |
| `EXPORT_ANNOTATION_IMAGES` | `True` | Saves the numbered frames I used for labelling. |

## Outputs

Everything goes to `/kaggle/working/seadronessee_mot_newvideos/<key>/`, and that folder is zipped at the end. `<key>` is a hash of the plan and settings; it was `eaafa4cd8dfb3eda3339` for my run. During extraction the features are also saved in `/kaggle/working/seadronessee_mot_feature_cache/`.

In the main folder:

- `experiment_plan.json`: plan, settings, model, annotation hashes
- `run_config.json`: plan, package versions, model commit, GPU and notes
- `README.txt`: short description of the files
- `sequence_overview.csv`: frames, boxes and box sizes per clip
- `descriptive_summary.csv`: IDs per setting and links changed vs `dino224_keep2`
- `appearance_contribution.csv`: links changed by adding DINOv3
- `constant_feature_controls.csv`: constant feature checks (all True)
- `changed_links.csv`: every changed link as `obs_id` pairs
- `identity_link_metrics.csv`: link P/R/F1 for labelled clips
- `feature_cache/features224_*.npz`: cached DINOv3 features

In each clip folder (`sat0131/` and `thu0109/`):

- `observations.csv`: one row per box (`obs_id`, `frame_box_key`, box, category)
- `selected_frames.csv`: chosen images with a SHA-256 per image
- `skipped_boxes.csv`
- `identity_labels_template.csv`: label sheet with an empty `gt_id` column
- `tracking_predictions.csv`: track ID per box for each setting
- `preview_first_frame.jpg`
- `tracking_comparison.mp4`: 6 panels, top row geometry only, bottom row with DINOv3, columns keep2, keep4 and damped keep4
- `video_export.log`
- `annotation_frames/fNNN.jpg`: frames with box indices `dNNN` for labelling

Cells 19 to 22 also write, inside `thu0109/`:

- `identity_review/identity_labels.csv`: a review sheet with a `confirmed` column.
- `automatic_comparison/`, also zipped as `thu0109_automatic_comparison.zip`:
  - from cells 20 and 21: `automatic_summary.csv`, `appearance_contribution.csv`, `changed_links.csv`, `tracking_predictions.csv`, `experiment_settings.csv`, `constant_feature_controls.csv`, `experiment_protocol.json`;
  - from cell 22: `identity_link_metrics.csv` (P/R/F1 for the eight settings), `dino_changes_checked.csv` (whether each added link was correct and each removed link was wrong) and `link_errors.csv` (wrong and missing links left in each method).


The follow up results are included in this repository as `thu0109_automatic_comparison.zip`. Cell 18 generates the full results archive, `SeaDronesSee_MOT_NewVideos_Results_<key>.zip`, in every run. I save that archive and attach it as a Kaggle input on later runs to reuse the cached DINOv3 features.

## Labelling

1. I open `annotation_frames/fNNN.jpg`. Each box shows `dNNN`, its index in that frame, which matches `frame_box_key = fNNN_dNNN` in the template.
2. I give the same `gt_id` to the same target in every frame and a different `gt_id` to different targets, starting from 1 in each clip.
3. I upload the filled sheet as an Input and set its path in `GT_ID_FILES`. The scoring only reads `obs_id` and `gt_id`. Every `obs_id` has to be there once, no `gt_id` can be empty, and two boxes in the same frame cannot share a `gt_id`.

The box index follows the annotation order, so it can change between frames. In `thu0109` frame `f013`, for example, `d001` is target 3 and `d002` is target 2.

## Results

Run `eaafa4cd8dfb3eda3339`, DINOv3 commit `114c137…`.

**`sat0131` (not labelled).** Adding DINOv3 changed **0** links in all three settings, and every setting made 9 IDs.

**`thu0109` (labelled, 147 true links):**

| Motion model | Gate | Geometry F1 | With DINOv3 F1 | IDs | Links changed by DINOv3 |
|---|---|---|---|---|---|
| Constant velocity | 0.12 | 0.941 | **0.955** | 8 | 4 |
| Last position | 0.12 | 0.927 | 0.927 | 8 | 0 |
| Constant velocity | 0.16 | **0.986** | **0.986** | 3 | 0 |
| Last position | 0.16 | 0.956 | 0.956 | 4 | 0 |

- Cell 22 writes all eight rows to `identity_link_metrics.csv` in `thu0109_automatic_comparison.zip`. The constant velocity, gate 0.12 row is also in the main `identity_link_metrics.csv` from cell 14, where `keep4` and `damped_keep4` get exactly the same scores as `keep2`.
- **Every change DINOv3 made was right.** In the one setting where it changed anything, it removed 2 wrong links and added 2 correct ones. This was two targets close together between samples 15 and 16 (`f014`→`f015`), which geometry alone had swapped.
- **A wider gate helped more than DINOv3.** With gate 0.16, F1 went up to 0.986 with or without DINOv3.
- **One error is left in both versions:** an ID swap between samples 6 and 7 (`f005`→`f006`).

So in my runs, DINOv3 rarely changes a match, but every change it made was right. The gate and the motion model matter more.

## Notes

- I used the ground truth boxes, so I only test the matching step.
- I labelled one clip, `thu0109`, with 147 true links. Across the eight settings the correct links go from 134 to 145, and DINOv3 changed 4 links in total.
- I added gate 0.16 and the last position model after looking at `thu0109`, and I scored them on the same clip.
- DINOv3 only compares boxes that pass the gate. When a box falls outside the gate, appearance cannot link it back.
- The gate also scales the distance part of the cost, so changing it changes the cost too.
- I use my own link F1 instead of HOTA or IDF1.
- I do not compensate for camera motion, and my frames are 15 source frames apart.
- DINOv3 is frozen and general purpose. Small boxes get upscaled a lot to reach 224×224.
- Both clips come from the `train` split of the detection set.

## Requirements

- A Kaggle notebook with a GPU (T4) for the first feature extraction. After that a CPU is enough.
- Python 3 with `numpy`, `pandas`, `scipy`, `pillow` and `packaging`. The notebook installs missing ones.
- `torch` (already on Kaggle), `transformers >= 4.56, < 6` (installed if missing) and `huggingface_hub`.
- `ffmpeg` for `tracking_comparison.mp4`.
- Versions in my run (from `run_config.json`): numpy 2.1.3, pandas 2.3.3, scipy 1.16.3, pillow 13.0.0, torch 2.11.0+cu130, transformers 5.20.2.
