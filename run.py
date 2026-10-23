"""Repository CLI; defaults to plan only."""
import os,sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from topic_experiment.runtime import main
if __name__=='__main__':main()
