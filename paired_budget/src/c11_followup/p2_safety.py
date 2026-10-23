"""Output-path, CPU-only and CPU-time safeguards for the paired comparison; nothing is read on import."""
from pathlib import Path
import hashlib
import os
import resource
import time

REPOSITORY = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path(os.environ.get('PAIRED_BUDGET_OUTPUT_ROOT', REPOSITORY/'results'/'generated'/'paired_budget')).resolve()


def check_output(path, *, create=False):
    path = Path(path).expanduser().resolve()
    if path == OUTPUT_ROOT or not path.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Output must be a subdirectory of the output root ' + str(OUTPUT_ROOT))
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Output directory must be absent or empty')
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def require_cpu():
    expected = {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise ValueError('Set CUDA_VISIBLE_DEVICES empty and OMP/OPENBLAS/MKL_NUM_THREADS=1')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CpuBudget:
    def __init__(self, seconds):
        if seconds <= 0 or seconds > 7200:
            raise ValueError('CPU limit must be in (0, 7200] seconds')
        self.started = time.process_time()
        self.seconds = float(seconds)
        # Hard safety backstop supplements the checkpoint that preserves partial output.
        used = resource.getrusage(resource.RUSAGE_SELF)
        hard = int(used.ru_utime + used.ru_stime + seconds) + 2
        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_CPU)
        if old_hard != resource.RLIM_INFINITY:
            hard = min(hard, old_hard)
        resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))

    def check(self):
        if time.process_time() - self.started >= self.seconds:
            raise TimeoutError('CPU quota exhausted; partial output retained; no hypothesis decision')
