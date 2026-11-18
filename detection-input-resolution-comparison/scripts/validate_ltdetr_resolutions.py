"""Evaluate one immutable exported detector at multiple input resolutions."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--source', type=Path)
parser.add_argument('--precision', choices=('bf16-mixed','fp16-mixed','fp32'), default='bf16-mixed')
parser.add_argument('--sizes', nargs='+', default=['1280','1536','1920'],
                    help='Square N or HEIGHTxWIDTH, e.g. 1536 or 1152x2048.')
args = parser.parse_args()


def parse_size(text):
    parts = [int(v) for v in str(text).lower().split('x')]
    if len(parts) == 1:
        parts = parts * 2
    if len(parts) != 2 or any(v <= 0 or v % 32 for v in parts):
        raise SystemExit(f'Invalid size {text!r}: use N or HEIGHTxWIDTH, multiples of 32')
    return tuple(parts)


SIZES = {str(s): parse_size(s) for s in args.sizes}
args.out.mkdir(parents=True, exist_ok=False)

import torch
import lightly_train
from lightly_train._commands import benchmark_task
from lightly_train._metrics.detection.task_metric import ObjectDetectionTaskMetric, ObjectDetectionTaskMetricArgs
from lightly_train._task_models.task_model_helpers import init_model_from_checkpoint


def classwise_metric(*, class_names):
    return ObjectDetectionTaskMetric(
        task_metric_args=ObjectDetectionTaskMetricArgs(classwise=True),
        split='val', class_names=class_names, box_format='xyxy', loss_names=[])


benchmark_task._create_metric = classwise_metric
base=Path(__file__).resolve().parents[1]/'runs/ltdetr_seadronessee'
source=args.source or base/'full_1280_frozen/exported_models/exported_best.pt'
snapshot=args.out/'source_best.pt'
shutil.copy2(source,snapshot)
sha=hashlib.sha256(snapshot.read_bytes()).hexdigest()
annotations=base/'data/instances_validation.json'
manifest={'source':str(source),'sha256':sha,'sizes':{k:list(v) for k,v in SIZES.items()},
          'annotations':str(annotations),'annotation_sha256':hashlib.sha256(annotations.read_bytes()).hexdigest(),
          'batch_size':1,'precision':args.precision,'threshold':0.0,
          'notes':'Validation only, original full images. Fresh model per size; identical source weights.'}
(args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
checkpoint=torch.load(snapshot,map_location='cpu',weights_only=False)
summary={}
for size, hw in SIZES.items():
    print(f'BEGIN VALIDATION {size}: same checkpoint SHA256 {sha}',flush=True)
    state=copy.deepcopy(checkpoint)
    state['model_init_args']['image_size']=hw
    state['model_init_args']['backbone_freeze']=True
    model=init_model_from_checkpoint(state,device='cuda')
    del state
    model.requires_grad_(False)
    assert tuple(model.image_size)==hw
    assert tuple(model.preprocessor.image_size)==hw
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    result=lightly_train.benchmark_object_detection(
        out=str(args.out/str(size)),dataset_name='SeaDronesSee source-group validation',
        model=model,data={'format':'coco','train':{'annotations':str(base/'data/instances_train.json')},
                         'val':{'annotations':str(annotations)}},
        batch_size=1,num_workers=4,threshold=0.0,warmup_steps=3,device='cuda',
        backend_args={'format':'torch','precision':args.precision})
    record=result.model_dump(mode='json')
    record['wall_seconds']=time.perf_counter()-started
    record['peak_gpu_memory_mib']=torch.cuda.max_memory_allocated()/1024**2
    summary[size]=record
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'COMPLETED {size}: {json.dumps(record["metric_values"])}',flush=True)
    result.print()
    del model,result
    gc.collect()
    torch.cuda.empty_cache()
print('All resolution validations completed; exiting to release GPU.',flush=True)
