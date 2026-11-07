"""Run only the eight missing arms, using original frozen code and inputs.

New output roots contain repaired cells only. Original studies are read-only.
The repair manifests identify the retained counterpart for later analysis.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile

sys.dont_write_bytecode = True
TASKS = {'path': (1, 3, 6, 7), 'control': (2, 5, 8, 13)}
ROOT = Path(__file__).resolve().parents[2]
CORE = 'scripts/circuit/path_patching.py'
CONTROL = 'scripts/circuit/path_patching_clean_control.py'
PLAN = 'results/aggregated/path_patching_setup/plan.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_gpu_names(names):
    if not names or any(re.search(r'\bH200\b', name) is None for name in names):
        raise ValueError(f'H200 GPUs required; actual visible devices: {names}')


def validate_task(kind, task_id):
    if task_id not in TASKS[kind]:
        raise ValueError(f'Not a missing {kind} repair task: {task_id}; allowed: {TASKS[kind]}')


def validate_output(output, source, controls):
    output = output.resolve()
    for original in (source.resolve(), controls.resolve()):
        if output == original or original in output.parents or output in original.parents:
            raise ValueError('Repair output must be separate from BOTH original studies')
    if (output/'study.json').exists() and not (output/'repair_manifest.json').exists():
        raise ValueError('Refusing an existing study without a repair manifest; choose a new output')


def copy_verified(source, target):
    if target.is_symlink():
        raise ValueError(f'Refusing output symlink: {target}')
    if target.exists():
        if sha(source) != sha(target):
            raise ValueError(f'Changed frozen input: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + '.tmp')
    shutil.copyfile(source, temp)
    temp.replace(target)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect(kind, task_id, source, controls):
    validate_task(kind, task_id)
    study = read(source/'study.json')
    controlstudy = read(controls/'study.json')
    if controlstudy['source_study_sha256'] != sha(source/'study.json'):
        raise ValueError('Control study refers to a different original study')
    if controlstudy['source_plan'] != study['plan'] or controlstudy['versions'] != study['versions']:
        raise ValueError('Original/control study settings mismatch')
    for p, digest in study['source_sha256'].items():
        if sha(source/'code_snapshot'/p) != digest:
            raise ValueError(f'Original snapshot changed: {p}')
    if sha(source/'code_snapshot'/PLAN) != study['plan_sha256']:
        raise ValueError('Frozen plan changed')
    if sha(controls/'code_snapshot'/CONTROL) != controlstudy['control_script_sha256']:
        raise ValueError('Executed clean-control script changed')
    control_module = load(controls/'code_snapshot'/CONTROL, 'original_clean_repair_input')
    core = control_module.load_core(source)
    _, task, jobs, source_hashes = control_module.inspect_task(source, task_id, core)
    cell = task['task']['cell']['id']
    counterfiles = [controls/'study.json', controls/cell/'runtime.json', controls/cell/'manifest.json']
    runtime = read(controls/cell/'runtime.json')
    for job in jobs:
        file = controls/cell/job['scope']/job['file'].name
        saved = read(file)
        if saved['pair'] != job['pair'] or saved['source_sha256'] != sha(job['file']):
            raise ValueError(f'Counterpart prompt/hash mismatch: {file}')
        if kind == 'path' and any(not r['passes_tolerance'] for r in saved['rows']):
            raise ValueError(f'Retained H200 control has failures: {file}')
        counterfiles.append(file)
    if kind == 'path':
        validate_gpu_names([runtime['gpu']])
    return {'cell': cell, 'task_id': task_id, 'kind': kind,
            'source_sha256': source_hashes,
            'control_source_sha256': {str(p.relative_to(controls)): sha(p) for p in counterfiles},
            'selected_heads': sum(len(g['sender_pairs']) for g in task['task']['groups']),
            'pairs': study['pairs_per_cell'], 'batch_size': study['batch_size']}


def freeze_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Changed repair settings/inputs: {path}')
    else:
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2, allow_nan=False))
        temp.replace(path)


def prepare(kind, task_id, source, controls, output, info):
    validate_output(output, source, controls)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.repair_prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        freeze_json(output/'repair_manifest.json', {
            'protocol': 'h200_missing_arms_v1', 'kind': kind, 'task_ids': list(TASKS[kind]),
            'source_root': str(source.resolve()), 'control_source_root': str(controls.resolve()),
            'source_study_sha256': sha(source/'study.json'),
            'control_study_sha256': sha(controls/'study.json'), 'gpu_family': 'H200',
            'driver_sha256': sha(Path(__file__)),
            'retain_other_cells': True,
            'join_rule': 'Use repaired arm only for listed tasks; retain original counterpart. Join by exact prompt, sender and receiver route, then verify new/retained clean margins. Old control source hashes refer to the original path rows, not the repaired rows.'})
        origin = (source if kind == 'path' else controls)/'code_snapshot'
        snapshot = output/'code_snapshot'
        if not snapshot.exists():
            temporary = Path(tempfile.mkdtemp(prefix='.repair_snapshot.', dir=output))
            shutil.copytree(origin, temporary, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            temporary.rename(snapshot)
        if snapshot.is_symlink():
            raise ValueError('Refusing symlink snapshot')
        if kind == 'path':
            for name, digest in read(source/'study.json')['source_sha256'].items():
                if sha(snapshot/name) != digest:
                    raise ValueError(f'Changed repair snapshot: {name}')
            copy_verified(source/'code_snapshot'/PLAN, snapshot/PLAN)
        elif sha(snapshot/CONTROL) != read(controls/'study.json')['control_script_sha256']:
            raise ValueError('Changed repair control snapshot')
        cell_dir = output/info['cell']
        if cell_dir.is_symlink():
            raise ValueError('Refusing symlink cell directory')
        cell_dir.mkdir(exist_ok=True)
        freeze_json(cell_dir/'repair_inputs.json', info)
        if kind == 'path':
            # Never copy old checkpoints: they would make the runner skip repair.
            # Copy exports byte-for-byte so execute_task cannot regenerate prompts.
            for name in ('pairs.jsonl', 'pairs.meta.json'):
                copy_verified(source/info['cell']/name, cell_dir/name)


def execute(args):
    source, controls, output = (Path(p).resolve() for p in
                               (args.source_root, args.control_source_root, args.output_root))
    validate_output(output, source, controls)
    info = inspect(args.kind, args.task_id, source, controls)
    print(json.dumps({k: v for k, v in info.items() if not k.endswith('sha256')}), flush=True)
    if args.preflight:
        return
    import torch
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    validate_gpu_names(gpu_names)
    original_core = load(source/'code_snapshot'/CORE, 'original_repair_core')
    versions = original_core.provenance()['versions']
    if versions != read(source/'study.json')['versions']:
        raise ValueError(f'Dependency versions changed: {versions}')
    prepare(args.kind, args.task_id, source, controls, output, info)
    with (output/f'.repair_task_{args.task_id}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = {'hostname': socket.gethostname(), 'gpu_names': gpu_names,
                   'versions': versions, 'driver_sha256': sha(Path(__file__)), 'slurm_job_id': os.getenv('SLURM_JOB_ID')}
        original_core.atomic_json(output/info['cell']/'repair_runtime.json', runtime)
        if args.kind == 'control':
            module = load(output/'code_snapshot'/CONTROL, 'frozen_clean_repair')
            module.execute(argparse.Namespace(source_root=str(source), output_root=str(output),
                task_id=args.task_id, tolerance=read(controls/'study.json')['tolerance'], preflight=False))
        else:
            # Bind the study's ordinary import to the byte-identical executed core.
            core = load(output/'code_snapshot'/CORE, 'frozen_path_repair_core')
            sys.modules['path_patching'] = core
            module = load(output/'code_snapshot/scripts/circuit/path_patching_study.py', 'frozen_path_repair_study')
            module.execute_task(argparse.Namespace(plan=str(output/'code_snapshot'/PLAN),
                task_id=args.task_id, pairs_per_cell=info['pairs'], batch_size=info['batch_size'],
                output_root=str(output), dry_run=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=list(TASKS), required=True)
    p.add_argument('--task-id', type=int, required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--control-source-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--preflight', action='store_true', help='Read-only audit; no CUDA requirement or output writes')
    execute(p.parse_args())

if __name__ == '__main__':
    main()
