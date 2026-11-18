# Few-shot maritime object classification — Weixun Ma

This experiment evaluates frozen DINOv3 features on five-class SeaDronesSee object-crop classification. Object locations are supplied by ground-truth bounding boxes; this is not a full detection or tracking pipeline.

## Files

- `dinov3_classification.ipynb`: executed Kaggle notebook, including saved numerical results and figures.
- `classification_full.py`: corresponding standalone code organized into five sections (expects Kaggle paths and attached inputs).
- `figures/`: figures extracted from the executed notebook.

## Data and pretrained weights

1. Attach the [SeaDronesSee dataset](https://www.kaggle.com/datasets/ubiratanfilho/sds-dataset) as a Kaggle Notebook input.
2. Obtain the web-image pretrained ViT-S/16 weights through the [official DINOv3 repository](https://github.com/facebookresearch/dinov3#pretrained-models), following its access and license requirements.
3. Attach `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` as a private Kaggle input. Weights and dataset images are not included here.

Expected data structure under the attached dataset:

```text
compressed/
  images/train/
  images/val/
  annotations/instances_train.json
  annotations/instances_val.json
```

## Running on Kaggle

1. Import the notebook and attach both inputs above.
2. Enable a GPU accelerator and Internet access. The experiment uses `cuda:0` only.
3. Run the five code cells in order.
4. Download `/kaggle/working/classification_full_results.zip` and save a Notebook version before ending the session.

The notebook searches `/kaggle/input` for the annotation file and weight filename. Attach only one matching copy of each. It clones the official DINOv3 repository into `/kaggle/working/dinov3`. The code uses PyTorch, torchvision, Pillow, NumPy, pandas, matplotlib, scikit-learn and tqdm from the Kaggle environment. It was run with PyTorch 2.11.0+cu130 on an NVIDIA Tesla T4. A different environment or repository revision may change results. Exact weight and annotation hashes and the repository commit are recorded in the generated `protocol.json`.

## Method

- Five classes: swimmer, boat, jetski, life-saving appliances and buoy.
- Ground-truth crops, directly resized to 256 x 256 without aspect-ratio preservation or padding.
- ImageNet normalization; frozen DINOv3 ViT-S/16; 384-dimensional CLS features.
- L2-normalize each feature, average reference features within each class, then L2-normalize each class prototype.
- Predict the class with the highest cosine similarity. No gradient-based classifier training or backbone fine-tuning.
- Compare 5, 20 and 100 reference objects per class, with seeds 42, 123 and 2026. Reference sets are nested within each seed.
- Evaluate all 9 settings on the same 9,630 validation objects with their natural class distribution.
- Extract features once for 1,450 unique selected training objects and all validation objects; cache completed batches.

## Results

Mean +/- sample standard deviation across three reference-sampling seeds:

| Examples/class | Accuracy (%) | Macro-F1 |
| --- | --- | --- |
| 5 | 65.05 +/- 10.01 | 0.5881 +/- 0.1019 |
| 20 | 78.66 +/- 6.80 | 0.6612 +/- 0.0485 |
| 100 | 79.45 +/- 0.38 | 0.6712 +/- 0.0006 |

Increasing the number of reference examples improved average performance and substantially reduced variation across the three sampled reference sets. Improvements were not monotonic for every seed.

At 20 examples/class, averaging over the three seeds, buoy classification accuracy was 36.2% for small, 83.9% for medium and 97.3% for large objects. Small life-saving appliances still achieved 94.4%. Large swimmers were a failure case (6.3% mean accuracy); in the fixed seed-42 run, 276 of 312 large swimmers were predicted as jetskis. This requires further analysis rather than assuming size alone explains performance.

Size groups use original bounding-box area: small < 32^2 pixels, medium >= 32^2 and < 96^2, large >= 96^2. Size refers to the original crop, not the resized model input.

## Outputs

The notebook creates metrics per run, summary statistics, class/size breakdowns, reference sample IDs, validation predictions, feature caches, a protocol record, four presentation figures and a ZIP archive. The main confusion matrix and illustrative errors use the predefined setting of 20 examples/class and seed 42; size heatmaps average all three seeds at 20 examples/class.

## Limitations

- Ground-truth boxes remove the localization problem; classification accuracy is not detection AP.
- Training and validation share 32 source videos. This does not establish generalization to unseen videos; crops can be correlated.
- Three seeds measure limited reference-sampling variability, not statistical confidence or independent-dataset replication.
- Validation labels are used for evaluation and exploratory error analysis, not prototype construction; no independent final test is reported.
- Direct square resizing can distort shapes, and enlarging tiny crops does not restore lost detail.
- The earlier balanced 500-object pilot uses a different evaluation distribution and reference draw, so its 77% accuracy is not directly comparable.

## References

- [DINOv3 official code and model access](https://github.com/facebookresearch/dinov3)
- [DINOv3 paper](https://arxiv.org/abs/2508.10104)
- [SeaDronesSee official repository](https://github.com/Ben93kie/SeaDronesSee)
- [Kaggle dataset used](https://www.kaggle.com/datasets/ubiratanfilho/sds-dataset)

The original DINOv3 source and pretrained weights are subject to the DINOv3 license. This folder does not redistribute them.
