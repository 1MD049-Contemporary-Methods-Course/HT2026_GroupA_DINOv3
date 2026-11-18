| Model / inference size | AP50 | AP | AP75 | AP50−AP75 | AP_small | Swimmer AP* | AR@100 | ms / img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 640 (separate 640 lineage)† | **73.8** | 46.3 | 48.9 | 24.9 | 23.3 | 33.9 | 62.7 | n/a |
| 1280-trained, infer 1280 | **74.9** | 47.6 | 50.2 | 24.7 | 24.8 | 35.8 | 65.2 | 22.0 |
| 1280-trained, infer 1536 (resize only) | **76.2** | 47.3 | 49.9 | 26.4 | 26.8 | 34.2 | 65.0 | 30.0 |
| 1280-trained, infer 1920 (resize only) | **73.0** | 45.1 | 48.1 | 24.8 | 25.3 | 32.5 | 64.1 | 46.4 |
| +2 ep at 1280, infer 1280 | **74.8** | 48.2 | 50.6 | 24.2 | 26.1 | 37.1 | 66.0 | 22.4 |
| +2 ep at 1536, infer 1536 | **77.3** | 49.3 | 51.8 | 25.5 | 28.0 | 37.3 | 66.0 | 30.0 |

\* Swimmer AP is AP@0.5:0.95; per-class AP50 was not logged and cannot be computed without re-running inference.
† 640 row: different training lineage and LightlyTrain native validation (no timing); not a controlled comparison.
All other rows: same 1446-image source-group validation set, independent benchmark, BF16, batch 1, L40S, score threshold 0.0.
"+2 ep" rows start from the same 1280 checkpoint with an identical recipe; only training size differs.
2048×1152 run omitted: it stopped after 1 of 2 epochs (epoch 1: AP50 76.0 vs 77.0 for 1536 at epoch 1; see CSV).
