# DINOv3 on SeaDronesSee

Group A · 1MD049 Contemporary Methods · Uppsala University · HT2026

We explored three uses of DINOv3 on the SeaDronesSee dataset: detecting objects in aerial images, retrieving objects of the same class, and tracking the same object across video frames.

**Dataset:** [`ubiratanfilho/sds-dataset`](https://www.kaggle.com/datasets/ubiratanfilho/sds-dataset) on Kaggle, a copy of the SeaDronesSee object detection set (images plus COCO-format `instances_train.json` / `instances_val.json`).

| Task | Contributor | What we tested | Files |
| --- | --- | --- | --- |
| Object detection | Paul Arnold (`P4ul-A`) | How do three DINOv3-based detectors compare on swimmers and boats, including a detector fine-tuned on SeaDronesSee? | [Detection notebook](sem_1_comparison.ipynb) |
| Detection input resolution | `nikochao` | Does training and running a frozen-DINOv3 LT-DETR detector at higher resolution (1280 → 1536) help it find small objects, and how much slower does it get? | [Results and scripts](detection-input-resolution-comparison/README.md) |
| Image retrieval | `chwa3055` | How well do DINOv3, DINOv2 and CLIP retrieve object crops of the same class as a query image? | [Retrieval notebook](DINOv3vsCLIP_retrieval.ipynb) |
| Multi-object tracking | Dan Zhang (`D11bot`) | Given the same bounding boxes, do frozen DINOv3 appearance features help a tracker keep the same ID on an object across frames? | [Tracking notebook](tracking/Tracking.ipynb) · [Method, manual labels and results](tracking/README.md) |
