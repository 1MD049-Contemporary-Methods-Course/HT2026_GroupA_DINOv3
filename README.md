# DINOv3 on SeaDronesSee

Group A · 1MD049 Contemporary Methods · Uppsala University · HT2026

We explored three uses of DINOv3 on the SeaDronesSee dataset: detecting objects in aerial images, retrieving objects of the same class, and tracking the same object across video frames.

| Task | Contributor | What we tested | Files |
| --- | --- | --- | --- |
| Object detection | Paul Arnold (`P4ul-A`) | How do three DINOv3-based detectors compare on swimmers and boats, including a detector fine-tuned on SeaDronesSee? | [Detection notebook](sem_1_comparison.ipynb) |
| Image retrieval | `chwa3055` | How well do DINOv3, DINOv2 and CLIP retrieve object crops of the same class as a query image? | [Retrieval notebook](DINOv3vsCLIP_retrieval.ipynb) |
| Multi-object tracking | Dan Zhang (`D11bot`) | Given the same bounding boxes, do frozen DINOv3 appearance features help a tracker keep the same ID on an object across frames? | [Tracking notebook](tracking/Tracking.ipynb) · [Method, manual labels and results](tracking/README.md) |
