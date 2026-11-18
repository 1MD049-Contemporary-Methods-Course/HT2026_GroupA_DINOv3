"""Evaluate identical epoch weights at 1280/1536 without changing training state."""
import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys

SIZES = ['1280', '1536']


def validate_epoch(out_dir, step, epoch, model_dict, logger):
    from lightly_train._commands import train_task_helpers as helpers
    directory = Path(out_dir)/'dual_validation'/f'epoch_{epoch:03d}_step_{step}'
    source_dir = directory/'weights'
    helpers.export_model(out_dir=source_dir, model_dict=model_dict, best_or_last='last')
    source = source_dir/'exported_models/exported_last.pt'
    script = Path(__file__).with_name('validate_ltdetr_resolutions.py')
    logger.info('Paired validation at epoch %d: same weights at %s.', epoch, ' and '.join(SIZES))
    subprocess.run([sys.executable, '-u', str(script), '--out', str(directory/'results'),
                    '--source', str(source), '--sizes', *SIZES,
                    '--precision', os.environ.get('LTDETR_EVAL_PRECISION', 'bf16-mixed')], check=True)
    logger.info('Paired validation complete: %s', directory/'results/summary.json')
    from ltdetr_validation_email import notify
    try:
        notify(directory/'results/summary.json', epoch, logger)
    except Exception:
        logger.exception('Validation notification failed; training results are preserved.')


def enable_dual_validation(max_epochs=None, sizes=None):
    if sizes:
        SIZES[:] = [str(s) for s in sizes]
    from lightly_train._commands import train_task
    from ltdetr_epoch_validation import adapt_epoch_validation
    from ltdetr_early_stopping import update
    original = train_task._train_task_from_config
    tree = adapt_epoch_validation(inspect.getsource(original))
    loop = [n for n in ast.walk(tree) if isinstance(n, ast.For)
            and isinstance(n.target, ast.Name) and n.target.id == 'step'][0]
    guard = loop.body[-1]
    if not isinstance(guard, ast.If):
        raise RuntimeError('Unexpected early-stopping hook structure')
    guard.body[:0] = ast.parse("""
_ltdetr_validate_epoch(out_dir, step, current_epoch, {
    'model_class_path': state['model_class_path'],
    'model_init_args': state['model_init_args'],
    'train_model': train_model.get_export_state_dict(),
    'license_info': state.get('license_info', ''),
}, logger)
""").body
    if max_epochs is not None:
        if max_epochs < 1:
            raise ValueError('max_epochs must be positive')
        guard.body.extend(ast.parse("""
if current_epoch >= _ltdetr_max_epochs:
    logger.info('Epoch budget reached (%d); saving and releasing GPU.', _ltdetr_max_epochs)
    helpers.save_checkpoint(fabric=fabric, out_dir=out_dir, state=state, best_or_last='last')
    helpers.export_model(out_dir=out_dir, model_dict={
        'model_class_path': state['model_class_path'],
        'model_init_args': state['model_init_args'],
        'train_model': train_model.get_export_state_dict(),
        'license_info': state.get('license_info', ''),
    }, best_or_last='last')
    break
""").body)
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace['_ltdetr_max_epochs'] = max_epochs
    namespace['_ltdetr_update_early_stopping'] = update
    namespace['_ltdetr_validate_epoch'] = validate_epoch
    exec(compile(tree, '<ltdetr-dual-validation>', 'exec'), namespace)
    train_task._train_task_from_config = namespace[original.__name__]
    print(f'Per-epoch paired {"/".join(SIZES)} validation enabled; early stopping uses native training-size mAP (patience=5, min_delta=0.005).', flush=True)
