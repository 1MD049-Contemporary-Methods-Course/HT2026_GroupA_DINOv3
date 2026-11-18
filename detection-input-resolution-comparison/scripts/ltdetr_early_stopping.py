"""Persist epoch-level mAP early stopping independently for each experiment."""
import json
import math
from pathlib import Path


def assess(history, min_delta=0.005, patience=5):
    anchor = None
    count = 0
    for record in history:
        score = record['map']
        if not math.isfinite(score):
            raise ValueError('Validation mAP is not finite.')
        if anchor is None or score - anchor >= min_delta - 1e-12:
            anchor, count = score, 0
        else:
            count += 1
    return anchor, count, count >= patience


def update(out_dir, step, epoch, score, logger):
    path = Path(out_dir) / 'early_stopping.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    # Discard records ahead of a restored checkpoint; count each epoch once.
    history = [r for r in previous.get('history', []) if r['step'] < step and r['epoch'] < epoch]
    history.append({'step': step, 'epoch': epoch, 'map': float(score)})
    anchor, count, stop = assess(history)
    state = {'metric': 'val_metric/map', 'min_delta': 0.005, 'patience': 5,
             'anchor': anchor, 'consecutive_without_improvement': count,
             'stopped': stop, 'history': history}
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2)+'\n')
    temp.replace(path)
    logger.info('Early stopping: mAP=%.6f, anchor=%.6f, patience=%d/5 (min_delta=0.005)', score, anchor, count)
    if stop:
        logger.info('Early stopping triggered after 5 epochs without qualifying improvement. Saving last checkpoint and stopping; best model is retained.')
    return stop
