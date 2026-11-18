"""Save full comparisons and queue per-validation Slurm summary emails.

No SMTP credentials are used. Slurm's END email contains the notification job
name (mAP and deltas); the full HTML/Markdown tables are stored beside results.
"""
import html
import json
from pathlib import Path
import subprocess

METRICS = {
    'mAP':'val_metric/map',
    'Small AP':'val_metric/map_small',
    'Swimmer AP':'val_metric_classwise/map_swimmer',
    'Life-saving AP':'val_metric_classwise/map_life_saving_appliances',
}


def build_report(current, previous, baseline, epoch):
    lines=[f'# Validation epoch {epoch}', '',
           'Values are percentages; differences are percentage points.', '',
           'FP16 continuation versus BF16 baseline: small numerical differences are possible.', '']
    titles=[]
    for size in ('1280','1536'):
        now=current[size]['metric_values']; before=previous[size]['metric_values']; base=baseline[size]['metric_values']
        delta=100*(now['val_metric/map']-before['val_metric/map'])
        gain=100*(now['val_metric/map']-base['val_metric/map'])
        label='improved' if delta > 0.0001 else 'regressed' if delta < -0.0001 else 'unchanged'
        titles.append(f'{size}:AP={100*now["val_metric/map"]:.2f},prev={delta:+.2f},base={gain:+.2f}pp({label})')
        lines += [f'## {size}: {label} versus previous validation','',
                  '| Metric | Before 1536 tuning | Previous | Current | Delta previous | Delta baseline |',
                  '|---|---:|---:|---:|---:|---:|']
        for title,key in METRICS.items():
            a,b,c=100*base[key],100*before[key],100*now[key]
            lines.append(f'| {title} | {a:.2f} | {b:.2f} | {c:.2f} | {c-b:+.2f} | {c-a:+.2f} |')
        lines.append('')
    return '\n'.join(lines), f'ValE{epoch} '+ ';'.join(titles)


def notify(summary_path, epoch, logger):
    summary_path=Path(summary_path).resolve()
    experiment=summary_path.parents[3]
    config_path=experiment/'email_notifications.json'
    if not config_path.exists():
        return
    config=json.loads(config_path.read_text())
    receipt=summary_path.with_name('email_notification.json')
    if receipt.exists():
        return
    current=json.loads(summary_path.read_text())
    candidates=[]
    # Include earlier L40S epochs when continuing into a new T4 directory.
    for root in [Path(config['previous_experiment']),experiment]:
        for path in (root/'dual_validation').glob('epoch_*_step_*/results/summary.json'):
            record_epoch=int(path.parents[1].name.split('_')[1])
            if record_epoch < epoch:
                data=json.loads(path.read_text())
                if all(size in data for size in ('1280','1536')):
                    candidates.append((record_epoch,path,data))
    if not candidates:
        raise RuntimeError('No earlier complete paired validation for comparison')
    previous=max(candidates,key=lambda item:item[0])[2]
    baseline=json.loads(Path(config['baseline_summary']).read_text())
    report,subject=build_report(current,previous,baseline,epoch)
    summary_path.with_name('comparison.md').write_text(report+'\n')
    summary_path.with_name('comparison.html').write_text('<html><body><pre>'+html.escape(report)+'</pre></body></html>\n')
    command=['sbatch','--parsable','--account=uppmax2026-1-299','--partition=pelle',
             '--ntasks=1','--cpus-per-task=1','--mem=128M','--time=00:01:00',
             '--job-name='+subject,'--mail-user='+config['recipient'],'--mail-type=END,FAIL',
             '--output='+str(summary_path.parent/'notification_%j.log'),'--wrap=/usr/bin/true']
    try:
        result=subprocess.run(command,check=True,capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.SubprocessError) as error:
        logger.error('Validation email could not be queued: %s',error)
        summary_path.with_name('email_error.txt').write_text(str(error)+'\n')
        return
    receipt.write_text(json.dumps({'recipient':config['recipient'],'summary':subject,
        'notification_job':result.stdout.strip(),'delivery':'Slurm END notification; full tables saved locally',
        'comparison_report':str(summary_path.with_name('comparison.md'))},indent=2)+'\n')
    logger.info('Validation summary email queued: %s',result.stdout.strip())
