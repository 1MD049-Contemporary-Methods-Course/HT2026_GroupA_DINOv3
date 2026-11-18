"""Train a frozen-backbone stage from the previous experiment's weights."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, choices=(640, 1280, 1536), required=True)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--steps', type=int, default=266112)
parser.add_argument('--out', type=Path)
parser.add_argument('--checkpoint', type=Path, help='Weights for a new training stage.')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--reuse-class-head', action='store_true', help='Keep the source checkpoint class head for the same dataset.')
parser.add_argument('--precision', choices=('bf16-mixed', '16-mixed', '32-true'), default='bf16-mixed')
parser.add_argument('--benchmark', action='store_true')
parser.add_argument('--small-object-crops', action='store_true')
parser.add_argument('--warmup-steps', type=int)
parser.add_argument('--dual-validation', action='store_true')
parser.add_argument('--max-epochs', type=int)
parser.add_argument('--hw', help='Non-square training size HEIGHTxWIDTH, e.g. 1152x2048; overrides --size.')
parser.add_argument('--train-annotations', type=Path, help='Alternative derived training COCO JSON.')
parser.add_argument('--eval-sizes', nargs='+', help='Dual-validation sizes (N or HEIGHTxWIDTH); default 1280 1536.')
args = parser.parse_args()
if args.hw is not None:
    try:
        hw = tuple(int(v) for v in args.hw.lower().split('x'))
    except ValueError:
        parser.error('--hw must be HEIGHTxWIDTH')
    if len(hw) != 2 or any(v <= 0 or v % 32 for v in hw):
        parser.error('--hw needs two positive multiples of 32')
    if args.out is None or args.checkpoint is None:
        parser.error('--hw requires a separate --out and explicit --checkpoint')
else:
    hw = (args.size, args.size)
if args.train_annotations is not None and not args.train_annotations.is_file():
    parser.error(f'Missing training annotations: {args.train_annotations}')
if args.max_epochs is not None and (not args.dual_validation or args.max_epochs < 1):
    parser.error('--max-epochs requires dual validation and a positive epoch count')
if not math.isfinite(args.lr) or args.lr <= 0:
    parser.error('Learning rate must be positive and finite')
if args.checkpoint is not None and args.out is None:
    parser.error('--checkpoint requires a separate --out directory')
if 32 % args.batch:
    parser.error('Batch must divide effective batch size 32')
import torch
import lightly_train

base = Path(__file__).resolve().parents[1] / 'runs/ltdetr_seadronessee'
source = base / ('full' if args.size == 640 else 'full_1280')
out = base / ('full_frozen' if args.size == 640 else 'full_1280_frozen')
out = args.out or out
continuation_config = out / 'continuation.json'
if continuation_config.exists():
    continuation = json.loads(continuation_config.read_text())
    args.max_epochs = continuation.get('max_epoch', args.max_epochs)
resume = (out / 'checkpoints/last.ckpt').is_file()
stop_state = out / 'early_stopping.json'
if not args.benchmark and stop_state.exists() and json.loads(stop_state.read_text()).get('stopped'):
    raise RuntimeError('This experiment already reached early stopping. Review early_stopping.json before explicitly starting a new stage.')
checkpoint = args.checkpoint or source / 'checkpoints/last.ckpt'
if checkpoint.resolve().parent.parent == out.resolve():
    parser.error('Source checkpoint and output must belong to different experiments')
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU allocation is required.')
transforms = {'image_size': hw}
if min(hw) >= 1280 or args.hw is not None:
    transforms['scale_jitter'] = None
if args.small_object_crops:
    if args.size != 640 or args.out is None:
        parser.error('Small-object crops require --size 640 and a separate --out')
    from ltdetr_small_object_crop import enable_small_object_crops
    enable_small_object_crops()
    transforms.update(random_iou_crop=None, random_zoom_out=None, scale_jitter=None,
                      mosaic=None, mixup=None, copyblend=None)
logger_args = {'log_every_num_steps': 10 if args.steps < 100 else 100}
if not args.benchmark:
    if args.dual_validation:
        from ltdetr_dual_validation import enable_dual_validation
        enable_dual_validation(max_epochs=args.max_epochs, sizes=args.eval_sizes)
    else:
        from ltdetr_epoch_validation import enable_epoch_validation
        enable_epoch_validation()
    train_annotations = args.train_annotations or base / 'data/instances_train.json'
    count = len(json.loads(train_annotations.read_text())['images'])
    logger_args['val_every_num_steps'] = max(1, math.ceil((count // args.batch) / (32 // args.batch)))
model_args = {'lr': args.lr, 'backbone_freeze': True}
if args.warmup_steps is not None:
    if args.warmup_steps < 0:
        parser.error('Warmup steps cannot be negative')
    model_args['lr_warmup_steps'] = args.warmup_steps
if args.size == 1536 and (args.out is None or args.checkpoint is None):
    parser.error('1536 requires a separate output and explicit source checkpoint')
train_annotations = args.train_annotations or base / 'data/instances_train.json'
print(f'Stage: checkpoint={checkpoint}, out={out}, image_size(HxW)={hw}, train={train_annotations}, lr={args.lr}, backbone_freeze=True, resume={resume}', flush=True)
lightly_train.train_object_detection(
    out=str(out), model='dinov3/vits16-ltdetr-coco',
    checkpoint=None if resume else str(checkpoint), resume_interrupted=resume,
    reuse_class_head=args.reuse_class_head,
    data={'format': 'coco', 'train': {'annotations': str(train_annotations)},
          'val': {'annotations': str(base / ('data/instances_smoke_val.json' if args.benchmark else 'data/instances_validation.json'))}},
    steps=args.steps, batch_size=args.batch, gradient_accumulation_steps=32 // args.batch,
    precision=args.precision, num_workers=args.workers, accelerator='auto', seed=88,
    transform_args=transforms, model_args=model_args,
    logger_args=logger_args,
    metric_args={'classwise': True, 'watch_metric': 'val_metric/map'},
    save_checkpoint_args={'save_best': not args.benchmark, 'save_last': not args.benchmark, 'save_every_num_steps': 100},
)
