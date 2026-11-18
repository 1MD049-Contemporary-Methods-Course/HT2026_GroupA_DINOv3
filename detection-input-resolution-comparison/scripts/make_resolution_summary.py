"""Collect existing resolution-experiment results into one CSV, a slide table and a plot.

Read-only: no training, no inference. Every number comes from an existing
summary.json (independent benchmark) or, for 640 runs, the native validation
table printed in train.log.
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'runs/ltdetr_seadronessee'
OUT = ROOT / 'reports/resolution_comparison'
CLASSES = ['swimmer', 'boat', 'jetski', 'life_saving_appliances', 'buoy']
FIELDS = (['row_id', 'group', 'train_size', 'infer_size', 'start_weights', 'extra_epochs', 'step',
           'eval_path', 'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'AR100', 'AR_small']
          + [f'AP_{c}' for c in CLASSES] + [f'AP50_{c}' for c in CLASSES]
          + ['latency_ms_median', 'peak_gpu_mib', 'gpu', 'precision', 'val_images', 'source', 'notes'])
KEYS = {'AP': 'val_metric/map', 'AP50': 'val_metric/map_50', 'AP75': 'val_metric/map_75',
        'AP_small': 'val_metric/map_small', 'AP_medium': 'val_metric/map_medium',
        'AP_large': 'val_metric/map_large', 'AR100': 'val_metric/mar_100', 'AR_small': 'val_metric/mar_small'}
KEYS.update({f'AP_{c}': f'val_metric_classwise/map_{c}' for c in CLASSES})
NA = 'not_available'


def from_metrics(m):
    row = {k: round(100 * m[v], 2) if v in m else NA for k, v in KEYS.items()}
    row.update({f'AP50_{c}': NA for c in CLASSES})
    return row


def benchmark_row(summary, size, **meta):
    path = Path(summary)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if size not in data:
        return None
    rec = data[size]
    row = from_metrics(rec['metric_values'])
    row.update(meta, infer_size=size, eval_path='independent benchmark (fresh model, batch 1, threshold 0.0)',
               latency_ms_median=round(1000 * rec['timing']['statistics']['latency_image_s']['median'], 1),
               peak_gpu_mib=round(rec['peak_gpu_memory_mib']) if 'peak_gpu_memory_mib' in rec else NA,
               gpu=rec['device_info']['gpu_name'], precision=rec['backend_args']['precision'],
               val_images=rec['num_images'], source=str(path.relative_to(ROOT)))
    return row


def native_row(experiment, **meta):
    exp = R / experiment
    best = max(json.loads((exp / 'early_stopping.json').read_text())['history'], key=lambda h: h['map'])
    lines = (exp / 'train.log').read_text(errors='replace').splitlines()
    metrics = None
    for i, line in enumerate(lines):
        if line.rstrip().endswith('Validation Metrics'):
            block = {}
            for nxt in lines[i + 1:i + 40]:
                cells = [c.strip() for c in nxt.split('|')]
                if len(cells) >= 5 and (cells[1].startswith('val_') or cells[1] == 'step'):
                    block[cells[1]] = float(cells[3])  # "Last" column = this validation
                elif block and not nxt.strip().startswith('|') and 'INFO]   |' not in nxt:
                    break
            if block.get('step') == best['step']:
                metrics = block
    if metrics is None:
        raise RuntimeError(f'No native validation block for step {best["step"]} in {experiment}')
    row = from_metrics(metrics)
    row.update(meta, step=best['step'], infer_size='640',
               eval_path='native validation during training (LightlyTrain), not the independent benchmark',
               latency_ms_median=NA, peak_gpu_mib=NA, gpu='NVIDIA L40S', precision='bf16-mixed',
               val_images=1446, source=str((exp / 'train.log').relative_to(ROOT)))
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp, name, note in [
            ('full_frozen', '640 frozen', 'Different lineage (640 pre-training with multi-scale aug, lr 1e-4); early-stopped; best epoch'),
            ('full_frozen_lr1e5', '640 frozen, lr 1e-5', 'Variant of 640; early-stopped; best epoch'),
            ('full_frozen_smallcrops_lr1e5', '640 small-object crops', 'Variant of 640 with object-centred crops; best epoch')]:
        rows.append(native_row(exp, row_id=exp, group='640 reference (separate lineage)', train_size='640',
                               start_weights='640 lineage', extra_epochs='early stopping', notes=note))
    base = R / 'validation_resolutions_7836374/summary.json'
    for s in ['1280', '1536', '1920']:
        rows.append(benchmark_row(base, s, row_id=f'base_infer{s}', group='inference-only resize',
                                  train_size='1280', start_weights='1280 frozen best.ckpt', extra_epochs=0, step=3001,
                                  notes='Same weights evaluated at each size; no training at that size'))
    runs = [('control1280', 'full_1280_frozen_lr1e5_2ep_control', '1280', ['1280', '1536']),
            ('ft1536', 'full_1536_frozen_lr1e5', '1536', ['1280', '1536']),
            ('ft2048x1152', 'full_2048x1152_frozen_lr1e5_2ep', '1152x2048', ['1152x2048', '1536'])]
    for rid, exp, train, sizes in runs:
        for epoch, step in [(1, 230), (2, 461)]:
            summary = R / exp / f'dual_validation/epoch_{epoch:03d}_step_{step}/results/summary.json'
            for s in sizes:
                note = 'Portrait training images rotated 90 deg' if rid == 'ft2048x1152' else ''
                row = benchmark_row(summary, s, row_id=f'{rid}_e{epoch}_infer{s}', group='fine-tuned at size (matched recipe)',
                                    train_size=train, start_weights='1280 frozen best.ckpt', extra_epochs=epoch, step=step, notes=note)
                rows.append(row or {'row_id': f'{rid}_e{epoch}_infer{s}', 'group': 'fine-tuned at size (matched recipe)',
                                    'train_size': train, 'infer_size': s, 'extra_epochs': epoch, 'step': step,
                                    'notes': 'NOT AVAILABLE: job 7871825 stopped manually after epoch 1; epoch 2 never ran'})
    e3 = R / 'full_1536_frozen_lr1e5_l40s/dual_validation/epoch_003_step_692/results/summary.json'
    for s in ['1280', '1536']:
        rows.append(benchmark_row(e3, s, row_id=f'ft1536_e3_infer{s}', group='1536 continuation (extra epoch)',
                                  train_size='1536', start_weights='1280 frozen best.ckpt', extra_epochs=3, step=692,
                                  notes='Third epoch from a resumed job; not part of the matched 2-epoch comparison'))
    rows.append(benchmark_row(ROOT / 'reports/ltdetr_ab_1920_7847726/A/evaluation/summary.json', '1920',
                              row_id='ft1536_e3_infer1920', group='1536 continuation (extra epoch)', train_size='1536',
                              start_weights='1280 frozen best.ckpt', extra_epochs=3, step=692,
                              notes='Same epoch-3 weights evaluated at 1920'))
    with (OUT / 'resolution_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval='')
        w.writeheader()
        w.writerows(r for r in rows if r)
    return rows


if __name__ == '__main__':
    for r in main():
        if r:
            print(r['row_id'], r.get('AP50'), r.get('AP'), r.get('AP_swimmer'), r.get('latency_ms_median'), r.get('notes', '')[:40])


SLIDE = [  # (row_id, label)
    ('full_frozen', '640 (separate 640 lineage)†'),
    ('base_infer1280', '1280-trained, infer 1280'),
    ('base_infer1536', '1280-trained, infer 1536 (resize only)'),
    ('base_infer1920', '1280-trained, infer 1920 (resize only)'),
    ('control1280_e2_infer1280', '+2 ep at 1280, infer 1280'),
    ('ft1536_e2_infer1536', '+2 ep at 1536, infer 1536'),
]


def fmt(v, d=1):
    try:
        return f'{float(v):.{d}f}'
    except (TypeError, ValueError):
        return 'n/a' if v == NA else 'pending'


def slide_and_plot():
    rows = {r['row_id']: r for r in csv.DictReader((OUT / 'resolution_results.csv').open())}
    lines = ['| Model / inference size | AP50 | AP | AP75 | AP50−AP75 | AP_small | Swimmer AP* | AR@100 | ms / img |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rid, label in SLIDE:
        r = rows.get(rid, {})
        gap = fmt(float(r['AP50']) - float(r['AP75'])) if r.get('AP50') not in (None, '', NA) else 'pending'
        lines.append(f"| {label} | **{fmt(r.get('AP50'))}** | {fmt(r.get('AP'))} | {fmt(r.get('AP75'))} | {gap} | "
                     f"{fmt(r.get('AP_small'))} | {fmt(r.get('AP_swimmer'))} | {fmt(r.get('AR100'))} | {fmt(r.get('latency_ms_median'))} |")
    lines += ['', '\\* Swimmer AP is AP@0.5:0.95; per-class AP50 was not logged and cannot be computed without re-running inference.',
              '† 640 row: different training lineage and LightlyTrain native validation (no timing); not a controlled comparison.',
              'All other rows: same 1446-image source-group validation set, independent benchmark, BF16, batch 1, L40S, score threshold 0.0.',
              '"+2 ep" rows start from the same 1280 checkpoint with an identical recipe; only training size differs.',
              '2048×1152 run omitted: it stopped after 1 of 2 epochs (epoch 1: AP50 76.0 vs 77.0 for 1536 at epoch 1; see CSV).']
    (OUT / 'slide_table.md').write_text('\n'.join(lines) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, grid = '#0b0b0b', '#52514e', '#e4e3df'
    blue, orange = '#2a78d6', '#eb6834'
    series = [('Resize at inference only (1280-trained model)', ['base_infer1280', 'base_infer1536', 'base_infer1920'],
               ['1280', '1536', '1920'], blue, 'o', '--'),
              ('Fine-tuned 2 epochs at that size', ['control1280_e2_infer1280', 'ft1536_e2_infer1536', 'ft2048x1152_e2_infer1152x2048'],
               ['1280', '1536', '2048×1152'], orange, 's', '-')]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
    for name, ids, labels, color, marker, ls in series:
        pts = [(float(rows[i]['latency_ms_median']), float(rows[i]['AP50']), lab) for i, lab in zip(ids, labels)
               if i in rows and rows[i].get('AP50') not in ('', None, NA)]
        xs, ys, ls_ = zip(*pts)
        ax.plot(xs, ys, ls, color=color, lw=2, marker=marker, ms=8, label=name,
                markeredgecolor='#fcfcfb', markeredgewidth=2, zorder=3)
        for x, y, lab in pts:
            dy = 9 if color == orange else -15
            ax.annotate(lab, (x, y), textcoords='offset points', xytext=(0, dy), ha='center', fontsize=9, color=muted)
    ax.set_xlabel('Inference time per image (ms, L40S, batch 1)', color=muted, fontsize=10)
    ax.set_ylabel('AP50 (%)', color=muted, fontsize=10)
    ax.set_title('AP50 vs. inference cost on SeaDronesSee validation', color=ink, fontsize=12, loc='left')
    ax.grid(True, color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=ink, loc='lower left')
    fig.tight_layout()
    fig.savefig(OUT / 'ap50_vs_inference_time.png', facecolor=fig.get_facecolor())


if __name__ == '__main__':
    slide_and_plot()
