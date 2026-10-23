"""Run smoke, poc or main configurations with provenance manifests and atomic output files."""
from __future__ import annotations
import argparse
import fcntl
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[2]

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def repository_files(repo=REPO):
    # Prune large evidence trees before walking; they have separate output hashes.
    excluded = {'archive', 'history', 'results', '.git', '__pycache__',
                '.pytest_cache', 'build', 'dist', 'provenance', 'audits'}
    files = []
    for parent, dirs, names in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in excluded and not d.startswith('.venv')
                   and not d.endswith('.egg-info')]
        for name in names:
            p = Path(parent)/name
            rel = p.relative_to(repo).as_posix()
            if (name == 'repository_manifest.json' or name.endswith(('.pyc', '.pyo', '.zip'))
                or rel.startswith(('data/raw/', 'manuscript/figures/', 'manuscript/tables/'))
                or rel == 'manuscript/manuscript.pdf'
                or (rel.startswith('manuscript/supplement/') and p.suffix in {'.json','.csv','.pdf'})):
                continue
            files.append(p)
    return sorted(files)

def source_hashes(repo=REPO):
    # Hash every repository file outside the excluded directories (code, configs, tests, documentation).
    return {p.relative_to(repo).as_posix(): sha(p) for p in repository_files(repo)}

def verify_repository(repo=REPO):
    record = json.loads((repo/'repository_manifest.json').read_text())
    actual = source_hashes(repo)
    rows = record['files']
    recorded = {row['path']: row['sha256'] for row in rows}
    if len(recorded) != len(rows) or recorded != actual:
        raise ValueError('Repository integrity mismatch: missing, additional or changed files')
    return dict(status='PASS', files=len(actual))

def verify_outputs(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('status') != 'PASS':
        raise ValueError('Run is not complete')
    if not {'rows.json','analysis.json'} <= set(manifest.get('output_hashes',{})):
        raise ValueError('Missing required output hashes')
    for name, expected in manifest['output_hashes'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError('Corrupt or unsafe output: ' + name)
    return manifest

def validate_config(config, repo=REPO):
    if config.get('stage') not in ('smoke','poc','main'):
        raise ValueError('Invalid stage')
    # An explicit 'experiment' key selects configs/{stage}_{experiment}.json as the canonical config; that file
    # must carry the same key, so experiment grids are validated exactly like the default configs.
    experiment = config.get('experiment')
    if experiment is not None and (not isinstance(experiment, str) or not experiment.isalnum()):
        raise ValueError('Invalid experiment name')
    name = config['stage'] if experiment is None else f"{config['stage']}_{experiment}"
    canonical = json.loads((repo/'configs'/f'{name}.json').read_text())
    if config['topic'] != canonical['topic'] or config['stage'] not in ('smoke', 'poc', 'main'):
        raise ValueError('Invalid topic/stage')
    seeds = config.get('seeds', [])
    if not seeds or len(seeds) != len(set(seeds)) or not set(seeds) <= set(canonical['seeds']):
        raise ValueError('Unknown, empty or duplicate stage seed')
    # Scientific parameters must equal the canonical config; only the keys in 'ignore' may differ.
    ignore = {'seeds', 'data', 'data_sha256', 'data_name', 'dataset_id', 'dataset_provenance', 'hypothesis_id', 'job_id'}
    for key in set(canonical) | set(config):
        if key in ignore:
            continue
        if config.get(key) != canonical.get(key):
            raise ValueError('Config differs from the canonical config: ' + key)
    if config.get('data'):
        if not config.get('data_sha256') or sha(config['data']) != config['data_sha256']:
            raise ValueError('Data checksum mismatch')
    if config['stage'] == 'main' and config.get('requires_data'):
        if not config.get('data') or not config.get('dataset_provenance'):
            raise ValueError('Main stage requires a verified input file and dataset provenance')
        if config['dataset_provenance'].get('confirmatory_eligible') is not True:
            raise ValueError('Dataset provenance must mark the input as eligible for the main stage')

def check_plan(config, plan_path, repo=REPO):
    if not plan_path:
        raise ValueError('Main execution disabled: --plan is required (create one with scripts/plan_main.py)')
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    for path, expected in plan.get('execution_sources', {}).items():
        execution_path = Path(path)
        if not execution_path.is_absolute():
            execution_path = repo/execution_path
        if sha(execution_path) != expected:
            raise ValueError('Execution source changed since the plan was created')
    jobs = [j for j in plan['jobs'] if j['id'] == config.get('job_id')]
    if len(jobs) != 1 or jobs[0]['config_hash'] != object_hash(config):
        raise ValueError('Job/config is not in the plan')
    if jobs[0]['source_hashes'] != source_hashes(repo):
        raise ValueError('Repository files changed after the plan was created')

def environment():
    versions = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'torch'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                versions=versions, cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})

def _execute(config, out, *, plan=None, execute_main=False, repo=REPO):
    if config.get('stage') == 'main':
        if not execute_main:
            raise ValueError('Main execution disabled: pass --execute-main together with --plan')
        check_plan(config, plan, repo)
    validate_config(config, repo)
    out = Path(out).resolve()
    if out.is_relative_to(repo.resolve()) and not out.is_relative_to((repo/'results/generated').resolve()):
        raise ValueError('In-repository runs must be under results/generated; archived results are immutable')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(status='RUNNING', stage=config['stage'], topic=config['topic'], config=config,
                    config_hash=object_hash(config), source_hashes=source_hashes(repo), environment=environment(),
                    started_utc=now(), pid=os.getpid(), process_identity=process_identity(), input_sha256=config.get('data_sha256'),
                    main_executed=config['stage']=='main', command=sys.argv)
    atomic_json(out/'manifest.json', manifest)
    try:
        from . import engine
        from .analysis import analyze
        rows = engine.run(config, out)
        if not rows:
            raise ValueError('Empty experiment output')
        atomic_json(out/'rows.json', rows)
        atomic_json(out/'analysis.json', analyze(config['topic'], rows))
        if manifest['source_hashes'] != source_hashes(repo):
            raise ValueError('Source changed during execution; result invalidated')
        manifest.update(status='PASS', rows=len(rows), finished_utc=now(), elapsed_seconds=time.monotonic()-started,
                        output_hashes={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'})
        atomic_json(out/'manifest.json', manifest)
        return manifest
    except BaseException as error:
        manifest.update(status='FAIL', finished_utc=now(), elapsed_seconds=time.monotonic()-started,
                        error_type=type(error).__name__, error=str(error))
        atomic_json(out/'manifest.json', manifest)
        raise

def process_identity():
    try:
        stat=Path(f'/proc/{os.getpid()}/stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+fields[19]
    except OSError:return None


def execute(config, out, *, plan=None, execute_main=False, repo=REPO):
    if config.get('stage')=='main':
        if not execute_main:raise ValueError('Main execution disabled: pass --execute-main together with --plan')
        check_plan(config,plan,repo)
    if config.get('job_id'):
        parent=Path(out).resolve().parent;parent.mkdir(parents=True,exist_ok=True)
        with (parent/'.execution.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('A worker already owns this job')
            return _execute(config,out,plan=plan,execute_main=execute_main,repo=repo)
    return _execute(config,out,plan=plan,execute_main=execute_main,repo=repo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['plan','smoke','poc','main','verify'], default='plan')
    parser.add_argument('--config'); parser.add_argument('--out'); parser.add_argument('--data')
    parser.add_argument('--plan'); parser.add_argument('--execute-main', action='store_true')
    args = parser.parse_args()
    if args.stage == 'plan':
        print((REPO/'configs/main.json').read_text()); return
    if args.stage == 'verify':
        print(json.dumps(verify_repository())); return
    if not args.out:
        parser.error('--out required')
    config = json.loads(Path(args.config or REPO/'configs'/f'{args.stage}.json').read_text())
    if config['stage'] != args.stage:
        parser.error('Stage and configuration disagree')
    if args.data:
        config.update(data=str(Path(args.data).resolve()), data_sha256=sha(args.data), data_name='provided_input')
    result = execute(config, args.out, plan=args.plan, execute_main=args.execute_main)
    print(json.dumps(dict(status=result['status'],topic=result['topic'],stage=result['stage'],rows=result['rows'])))

if __name__ == '__main__':
    main()
