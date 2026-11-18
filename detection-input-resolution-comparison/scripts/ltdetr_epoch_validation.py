"""Run validation at actual LightlyTrain epoch boundaries.

LightlyTrain 0.18 exposes only a fixed step interval. Adapt the validation
predicate in memory for this process, keeping its training/validation loop and
checkpoint handling intact. Refuse to patch if the upstream predicate changes.
No installed package files are modified.
"""
import ast
import inspect


def adapt_epoch_validation(source):
    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'is_val_step' for t in node.targets):
            matches.append(node)
    expected = ast.parse('(step + 1) % val_every_num_steps == 0', mode='eval').body
    if len(matches) != 1 or ast.dump(matches[0].value) != ast.dump(expected):
        raise RuntimeError('LightlyTrain validation loop changed; inspect before enabling per-epoch validation.')
    matches[0].value = ast.parse(
        'current_epoch > helpers.get_training_epoch(step=step - 1, '
        'train_num_batches=train_num_batches, '
        'gradient_accumulation_steps=config.gradient_accumulation_steps)',
        mode='eval',
    ).body
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step']
    if len(loops) != 1 or not any(isinstance(n, ast.Name) and n.id == 'val_agg_metric_values' for n in ast.walk(loops[0])):
        raise RuntimeError('LightlyTrain training loop changed; cannot install early stopping.')
    loops[0].body.extend(ast.parse("""
if is_val_step and val_agg_metric_values is not None:
    if _ltdetr_update_early_stopping(out_dir, step, current_epoch, val_agg_metric_values.metric_values['val_metric/map'], logger):
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
    return tree


def enable_epoch_validation():
    from lightly_train._commands import train_task
    original = train_task._train_task_from_config
    tree = adapt_epoch_validation(inspect.getsource(original))
    from ltdetr_early_stopping import update
    namespace = dict(original.__globals__)
    namespace["_ltdetr_update_early_stopping"] = update
    exec(compile(tree, '<ltdetr-per-epoch-validation>', 'exec'), namespace)
    train_task._train_task_from_config = namespace[original.__name__]
    print('Early stopping enabled: patience=5 epochs, min_delta=0.005 mAP.', flush=True)
    print('Validation schedule: once per completed epoch (after the optimizer step crossing its boundary), plus final-step validation.', flush=True)
