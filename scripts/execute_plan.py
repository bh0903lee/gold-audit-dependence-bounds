"""Run every job of a run plan through run.py with a bounded worker pool; CPU only.

Main-stage jobs are started as run.py --execute-main --plan PLAN, which re-checks the plan; this script only
schedules them.  Failed attempts are kept; a rerun adds attempt_NN next to them and skips jobs that already
have a PASS attempt.
"""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from topic_experiment import runtime as r
MAX_WORKERS=24

def status(path):
 try:return json.loads((path/'manifest.json').read_text()).get('status')
 except (OSError,ValueError):return None

def run_job(job,a,env):
 base=a.out/'jobs'/job['id'];base.mkdir(parents=True,exist_ok=True)
 attempts=sorted(p for p in base.glob('attempt_*') if p.is_dir())
 if any(status(p)=='PASS' for p in attempts):return dict(id=job['id'],status='PASS',attempt=None,skipped=True)
 out=base/f'attempt_{len(attempts)+1:02d}';config=a.out/'configs'/f"{job['id']}.json"
 if not config.exists():r.atomic_json(config,job['config'])
 cmd=[sys.executable,str(ROOT/'run.py'),'--stage',job['config']['stage'],'--config',str(config),'--out',str(out)]
 if job['config']['stage']=='main':cmd+=['--execute-main','--plan',str(a.plan)]
 started=time.monotonic()
 with (a.out/'logs'/f"{job['id']}.{out.name}.log").open('w') as log:code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
 return dict(id=job['id'],status=status(out) or 'NO_MANIFEST',returncode=code,attempt=out.name,seconds=time.monotonic()-started,skipped=False)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=MAX_WORKERS);a=p.parse_args()
 if not 1<=a.workers<=MAX_WORKERS:p.error(f'workers must be in 1..{MAX_WORKERS}')
 a.out=a.out.resolve();a.plan=a.plan.resolve()
 if not a.out.is_relative_to((ROOT/'results/generated').resolve()):p.error('--out must be under results/generated')
 plan=json.loads(a.plan.read_text());jobs=plan['jobs']
 for d in ('jobs','configs','logs'):(a.out/d).mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 started_utc=r.now();started=time.monotonic()
 with ThreadPoolExecutor(a.workers) as pool:results=list(pool.map(lambda j:run_job(j,a,env),jobs))
 record=dict(plan=str(a.plan),plan_sha256=r.sha(a.plan),workers=a.workers,started_utc=started_utc,finished_utc=r.now(),wall_seconds=time.monotonic()-started,
  jobs=len(jobs),passed=sum(x['status']=='PASS' for x in results),failed=[x for x in results if x['status']!='PASS'],cpu_seconds_in_jobs=sum(x.get('seconds',0.) for x in results),results=results)
 n=len(list(a.out.glob('execution_*.json')));r.atomic_json(a.out/f'execution_{n+1:02d}.json',record)
 print(json.dumps({k:record[k] for k in ('jobs','passed','wall_seconds','workers')}|{'failed':len(record['failed'])}));return int(record['passed']!=len(jobs))
if __name__=='__main__':sys.exit(main())
