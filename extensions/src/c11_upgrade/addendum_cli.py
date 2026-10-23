"""Command-line runner for the overlapping-window experiment (e8-full), the general-law experiment (e9-full) and the recorded-occupancy application (e7-full). Nothing runs on import."""
from __future__ import annotations
import os
# Apply CPU-only controls before importing numerical libraries (also under -m).
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get('EXTENSIONS_OUTPUT_ROOT', ROOT.parent/'results'/'generated'/'extensions')).resolve()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def validate_config():
    config = json.loads((ROOT/'configs/addendum_e7_e9.json').read_text())
    for name,digest in config['baseline_hashes'].items():
        if sha256(ROOT/name) != digest: raise ValueError(f'Protected baseline changed: {name}')
    return config


def output_path(path):
    candidate = Path(path).expanduser().resolve()
    root = OUTPUT_ROOT.resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f'Output must be a new subdirectory of the output root {root}')
    if candidate.exists() and (not candidate.is_dir() or any(candidate.iterdir())):
        raise ValueError('Output directory is not empty')
    return candidate


def start_run(output, kind):
    config = validate_config()
    out = output_path(output)
    out.mkdir(parents=True, exist_ok=True)
    files = list((ROOT/'src/c11_upgrade').glob('*.py')) + list((ROOT/'tests').glob('*.py'))
    files += [ROOT/'configs/addendum_e7_e9.json',ROOT/'configs/protocol.json']
    import numpy, scipy
    manifest = dict(kind=kind, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    scientific_run=kind in ('e8_full','e9_full','e7_full'),run_complete=False,
                    hashes={str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(files))},
                    python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,platform=platform.platform(),
                    environment={k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    write_json(out/'manifest.json',manifest)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return out,config,(time.monotonic(),usage.ru_utime+usage.ru_stime)


def quota(start, out, config):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Cooperative between LP/replicate calls, not a hard interrupt of HiGHS.
    reasons = []
    if time.monotonic()-start[0] > config['max_wall_seconds']: reasons.append('wall')
    if usage.ru_utime+usage.ru_stime-start[1] > config['max_cpu_seconds']: reasons.append('cpu')
    if usage.ru_maxrss > config['max_memory_gib']*1024**2: reasons.append('memory')
    if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > config['max_output_gib']*1024**3: reasons.append('output')
    return reasons


def finish(out, summary, complete=False):
    write_json(out/'summary.json',summary)
    path = out/'manifest.json'; manifest = json.loads(path.read_text())
    manifest['run_complete'] = bool(complete)
    manifest['output_hashes'] = {p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
    write_json(path,manifest)
    return summary


def plan(output):
    out,config,_ = start_run(output,'plan_only')
    summary = dict(scientific_verdict='NOT_EVALUATED',executed_replicates=0,
        e8=dict(conditions=7,replicates=7000,methods=6,intervals=42000,endpoint_lps=84000),
        e9=dict(laws=80,budgets=3,replicates=48000,information_levels=3,intervals=144000,endpoint_lps=288000),
        e7=dict(status='DATA_AND_E8_SELECTION_PENDING',max_replicates_per_lag=3000,lag_multipliers=[1,5]),
        resource_limits={k:config[k] for k in ('workers','max_cpu_seconds','max_wall_seconds','max_memory_gib','max_output_gib')},
        quota_caveat='Cooperative checks between calls; no single-LP hard interrupt')
    return finish(out,summary,False)


def e8_full(output):
    from . import e8
    from .population import markov_law
    from .addendum_core import population_references
    out,config,start = start_run(output,'e8_full'); rows=[]; stopped=[]
    with (out/'rows.jsonl').open('w') as stream:
        for condition in e8.condition_grid():
            args = {k:v for k,v in condition.items() if k in ('stay','detect','phi','prevalence','rho')}
            reference = population_references(markov_law(**args))
            for seed in range(440001,441001):
                stopped = quota(start,out,config)
                if stopped: break
                rep = e8.run_replicate(condition,seed,reference)
                for row in rep['metrics']:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                    rows.append({k:v for k,v in row.items() if k not in ('diagnostics','calibration','limits')})
                stream.flush()
            if stopped: break
    summary = e8.summarize(rows,smoke=False)
    stopped = stopped or quota(start,out,config)
    summary['quota_stop'] = stopped
    if stopped:
        summary['h8a'] = 'incomplete_resource_limit'
        summary['h8b'] = {m:'incomplete_resource_limit' for m in ('thin_10','block_200')}
        summary['selection'] = dict(method='application_only',reason='incomplete_resource_limit',coverage_certification=False)
    return finish(out,summary,summary['complete_grid'] and not stopped)


def e9_full(output):
    from . import e9
    from .population import periodic_law
    from .addendum_core import population_references
    out,config,start = start_run(output,'e9_full'); rows=[]; stopped=[]
    with (out/'rows.jsonl').open('w') as stream:
        for spec in e9.law_specs():
            reference = population_references(periodic_law(spec['law_seed'],alphabet=3 if spec['bank']=='A' else 4))
            for budget in (200,500,2000):
                for seed in range(450001,450201):
                    stopped = quota(start,out,config)
                    if stopped: break
                    rep = e9.run_replicate(spec,budget,seed,confirmatory=True,reference=reference)
                    for row in rep['metrics']:
                        stream.write(json.dumps(row,allow_nan=False)+'\n'); rows.append(row)
                    stream.flush()
                if stopped: break
            if stopped: break
    summary = e9.summarize(rows,confirmatory=not bool(stopped))
    stopped = stopped or quota(start,out,config)
    summary['quota_stop'] = stopped
    if stopped:
        summary['resource_status']='incomplete_resource_limit'
        summary['h9b']='BLOCKED_resource_limit'
        summary['h9a']='NOT_EVALUATED'
    return finish(out,summary,summary.get('full_grid_complete',False) and not stopped)


def e8_selection(path):
    """Verify a completed e8-full run (manifest, row and summary hashes) and recompute its method selection from the raw rows."""
    config = validate_config()
    run = Path(path).resolve(); root = OUTPUT_ROOT.resolve()
    if run == root or not run.is_relative_to(root):
        raise ValueError('Selection input must be a subdirectory of the output root')
    for name in ('manifest.json','rows.jsonl','summary.json'):
        if (run/name).resolve().parent != run:
            raise ValueError('Run files must be regular files inside the run directory, not links')
    manifest = json.loads((run/'manifest.json').read_text())
    if manifest.get('kind')!='e8_full' or manifest.get('run_complete') is not True:
        raise ValueError('A complete e8-full run is required')
    required = ('rows.jsonl','summary.json')
    for name in required:
        if manifest.get('output_hashes',{}).get(name) != sha256(run/name):
            raise ValueError('e8-full output bytes differ from its manifest')
    required_sources = set(config['baseline_hashes']) | {
        'src/c11_upgrade/e8.py','src/c11_upgrade/addendum_core.py',
        'src/c11_upgrade/addendum_cli.py','configs/addendum_e7_e9.json'}
    for name in required_sources:
        if manifest.get('hashes',{}).get(name) != sha256(ROOT/name):
            raise ValueError('e8-full run was produced by different source or config files')
    from .e8 import summarize
    rows=[]
    with (run/'rows.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            rows.append({k:v for k,v in row.items() if k not in ('diagnostics','calibration','limits')})
    checked = summarize(rows,smoke=False,bootstrap_reps=1)  # CI for selection is exact CP, not bootstrap.
    supplied = json.loads((run/'summary.json').read_text())
    if not checked['complete_grid'] or checked['selection'] != supplied.get('selection'):
        raise ValueError('Method selection does not reproduce from the raw rows')
    return dict(method=checked['selection']['method'],e8_run=str(run.relative_to(root)),manifest_sha256=sha256(run/'manifest.json'),
                summary_sha256=sha256(run/'summary.json'),coverage_certification=False)


def e7_full(output, source, manifest_path, selection_path=None, application_only=False):
    from . import e7
    if bool(selection_path) == bool(application_only):
        raise ValueError('Pass exactly one of --e8-run or --application-only')
    selection = (dict(method='application_only',reason='explicit_descriptive_only',coverage_certification=False)
                 if application_only else e8_selection(selection_path))
    source = Path(source); contract = json.loads(Path(manifest_path).read_text())
    e7.validate_manifest(source,contract)
    out,config,start = start_run(output,'e7_full')
    write_json(out/'input_manifest.json',contract);write_json(out/'selection.json',selection)
    summaries=[];stopped=[]
    with (out/'rows.jsonl').open('w') as stream:
        for multiplier in (1,5):
            frame = e7.prepare_frame(source,contract,selection['method'],multiplier)
            if frame['status']!='ready':
                summaries.append(dict(lag_multiplier=multiplier,status='insufficient_population',groups=[]));continue
            reference = e7.empirical_reference(frame['frames'])
            gate = e7.gate_probe(frame['frames'],contract,multiplier);gate['source_hash_verified']=True
            if gate['status'].startswith('blocked'):
                summaries.append(dict(lag_multiplier=multiplier,status='blocked_gate',gate=gate));continue
            rows=[]
            for budget in (50,200,500):
                if len(frame['frames'])<2*budget: continue
                for seed in range(460001,461001):
                    stopped=quota(start,out,config)
                    if stopped: break
                    rep=e7.run_replicate(frame,budget,seed,reference)
                    stream.write(json.dumps(rep,allow_nan=False)+'\n');stream.flush()
                    rows.extend(rep['metrics'])
                if stopped: break
            summary=e7.summarize(rows,frame,reference,gate,confirmatory=not bool(stopped))
            summary.update(lag_multiplier=multiplier,excluded=frame['excluded'],source_sha256=frame['source_sha256'])
            summaries.append(summary)
            if stopped: break
    stopped=stopped or quota(start,out,config)
    if stopped:
        for summary in summaries: summary['application_criterion']='not_evaluable_resource_limit'
    return finish(out,dict(inference_scope='finite_population_descriptive',lags=summaries,quota_stop=stopped,
                          selection=selection),not stopped and len(summaries)==2 and
                  all(s.get('complete_eligible_grid',False) for s in summaries))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('plan','e8-full','e9-full','e7-full'):
        p=sub.add_parser(name);p.add_argument('--output',required=True)
        if name=='e7-full':
            p.add_argument('--source',required=True);p.add_argument('--manifest',required=True)
            group=p.add_mutually_exclusive_group(required=True)
            group.add_argument('--e8-run');group.add_argument('--application-only',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='plan': result=plan(args.output)
        elif args.command=='e8-full': result=e8_full(args.output)
        elif args.command=='e9-full': result=e9_full(args.output)
        else: result=e7_full(args.output,args.source,args.manifest,args.e8_run,args.application_only)
        print(json.dumps(dict(command=args.command,output=str(Path(args.output).resolve()),
                              scientific_run=args.command!='plan',status='completed_cli'),ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        return 1

if __name__=='__main__': raise SystemExit(main())
