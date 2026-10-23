"""Run the three unit-test suites (core, extensions, paired_budget) offline on the CPU; exit code is the maximum."""
import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
suites=[('core',root/'tests',None),
        ('extensions',root/'extensions'/'tests',[root/'extensions'/'src']),
        ('paired_budget',root/'paired_budget'/'tests',[root/'paired_budget'/'src',root/'extensions'/'src'])]
codes=[]
for name,tests,path in suites:
    env=dict(base)
    if path:env['PYTHONPATH']=os.pathsep.join(str(p) for p in path)
    print(f'== {name} ==',flush=True)
    codes.append(subprocess.call([sys.executable,'-B','-m','unittest','discover','-s',str(tests),'-t',str(tests),'-v'],cwd=root,env=env))
raise SystemExit(max(codes))
