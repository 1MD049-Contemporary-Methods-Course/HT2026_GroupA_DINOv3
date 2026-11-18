"""Standard pycocotools COCOeval on saved detections; reports overall and per-class AP50/AP. CPU only."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(annotations, detections):
    gt_data = json.loads(Path(annotations).read_text())
    gt_data = copy.deepcopy(gt_data)
    for a in gt_data['annotations']:
        a.setdefault('iscrowd', 0)  # derived JSON has no crowd boxes; pycocotools needs the field
    gt = COCO()
    gt.dataset = gt_data
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(str(detections))
        ev = COCOeval(gt, dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
    names = {c['id']: c['name'] for c in gt_data['categories']}
    p = ev.eval['precision']  # [T, R, K, A, M]; A=0 all areas, M=2 maxDets=100

    def ap(t_slice, k):
        v = p[t_slice, :, k, 0, 2]
        v = v[v > -1]
        return float(np.mean(v)) * 100 if v.size else float('nan')

    out = {'overall': {'AP': ev.stats[0] * 100, 'AP50': ev.stats[1] * 100, 'AP75': ev.stats[2] * 100,
                       'AP_small': ev.stats[3] * 100, 'AR100': ev.stats[8] * 100}}
    for k, cid in enumerate(ev.params.catIds):
        out[names[cid]] = {'AP50': ap(0, k), 'AP75': ap(5, k), 'AP': ap(slice(None), k)}
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, default=Path('runs/ltdetr_seadronessee/data/instances_validation.json'))
    args = parser.parse_args()
    result = evaluate(args.annotations, args.detections)
    (args.detections.parent / 'coco_per_class.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
