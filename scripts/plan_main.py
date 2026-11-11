"""Create a main-stage run plan (one job per seed). This command does not run the main stage."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from topic_experiment import runtime as r

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--experiment',help='canonical configs/main_{experiment}.json instead of configs/main.json');a=p.parse_args()
 r.verify_repository()
 # The plan binds every hashed repository file, so it must not become one itself (results/ is not hashed).
 out=a.out.resolve()
 if out.is_relative_to(ROOT.resolve()) and not out.is_relative_to((ROOT/'results').resolve()):p.error('Write the plan under results/ (e.g. results/plans/main.json) or outside the package')
 if a.out.exists():p.error('Refusing to overwrite an existing plan')
 name='main' if not a.experiment else f'main_{a.experiment}'
 config=json.loads((ROOT/'configs'/f'{name}.json').read_text());seeds=a.seeds or config['seeds']
 if config.get('experiment')!=a.experiment:p.error('Canonical config and --experiment disagree')
 prefix='C11-seed-' if not a.experiment else f'C11-{a.experiment}-seed-'
 if len(set(seeds))!=len(seeds) or not set(seeds)<=set(config['seeds']):p.error('Unknown or duplicate seeds')
 hashes=r.source_hashes();jobs=[]
 for seed in seeds:
  c=dict(config,seeds=[seed],job_id=f'{prefix}{seed}');r.validate_config(c)
  jobs.append(dict(id=c['job_id'],topic='C11',config=c,config_hash=r.object_hash(c),source_hashes=hashes))
 plan=dict(version=1,topic='C11',created_utc=r.now(),path_base='repository root',jobs=jobs,execution_sources={k:hashes[k] for k in ['run.py','scripts/plan_main.py','src/topic_experiment/runtime.py']})
 r.atomic_json(a.out,plan)
 print(json.dumps({'plan':str(a.out),'sha256':r.sha(a.out),'jobs':len(jobs),'main_executed':False}))
if __name__=='__main__':main()
