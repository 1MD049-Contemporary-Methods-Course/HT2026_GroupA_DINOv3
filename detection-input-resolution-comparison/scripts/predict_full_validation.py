"""Save every validation prediction (score threshold 0) in COCO results format.

Inference only: immutable weights, no training. Output feeds pycocotools offline.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from lightly_train._task_models.task_model_helpers import init_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--annotations', type=Path,
                        default=Path('runs/ltdetr_seadronessee/data/instances_validation.json'))
    args = parser.parse_args()
    if args.size <= 0 or args.size % 32:
        parser.error('--size must be a positive multiple of 32')
    args.out.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.annotations.read_text())
    cat_id = {c['name']: c['id'] for c in data['categories']}
    state = torch.load(args.source, map_location='cpu', weights_only=False)
    state['model_init_args']['image_size'] = (args.size, args.size)
    model = init_model_from_checkpoint(state, device='cuda').eval()
    del state
    model.requires_grad_(False)
    assert tuple(model.image_size) == (args.size, args.size)
    assert tuple(model.preprocessor.image_size) == (args.size, args.size)
    assert set(model.classes.values() if isinstance(model.classes, dict) else model.classes) == set(cat_id), model.classes
    manifest = dict(source=str(args.source.resolve()), sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    annotations=str(args.annotations.resolve()),
                    annotations_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                    images=len(data['images']), image_size=args.size, score_threshold=0.0,
                    precision='bf16-mixed autocast', box_format='COCO xywh, original image pixels',
                    gpu=torch.cuda.get_device_name())
    results = []
    started = time.perf_counter()
    for n, info in enumerate(data['images'], 1):
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model.predict(info['file_name'], threshold=0.0)
        boxes = pred['bboxes'].detach().float().cpu().tolist()
        scores = pred['scores'].detach().float().cpu().tolist()
        labels = pred['labels'].detach().cpu().tolist()
        for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
            results.append(dict(image_id=info['id'], category_id=cat_id[model.classes[int(l)]],
                                bbox=[round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)], score=round(s, 6)))
        if n % 200 == 0:
            print(f'{n}/{len(data["images"])} images, {len(results)} detections', flush=True)
    manifest['detections'] = len(results)
    manifest['wall_seconds'] = round(time.perf_counter() - started, 1)
    (args.out / 'detections.json').write_text(json.dumps(results))
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(results)} detections for {len(data["images"])} images to {args.out}', flush=True)


if __name__ == '__main__':
    main()
