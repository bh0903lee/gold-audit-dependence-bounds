"""Shared loaders and statistics for the six-variable comparison and the assumption-violation study."""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/"src"))  # package root (three levels above this directory)
from topic_experiment import runtime

Z = 1.95996398454


def wilson(k, n):
    if n == 0: return None
    p = k/n; centre = (p + Z*Z/(2*n))/(1 + Z*Z/n); rad = Z*np.sqrt(p*(1 - p)/n + Z*Z/(4*n*n))/(1 + Z*Z/n)
    return [float(centre - rad), float(centre + rad)]


def mean_t(values):
    """Mean and Student-t 95% Monte Carlo interval over seeds."""
    v = np.asarray(values, float); n = len(v)
    if n == 0: return dict(n=0, mean=None, interval=None)
    if n < 2 or v.std(ddof=1) == 0: return dict(n=n, mean=float(v.mean()), interval=[float(v.mean())]*2 if n > 1 else None)
    half = stats.t.ppf(.975, n - 1)*v.std(ddof=1)/np.sqrt(n)
    return dict(n=n, mean=float(v.mean()), interval=[float(v.mean() - half), float(v.mean() + half)])


def load_run(root, plan=None):
    """Rows of an executor run directory. Every planned job must have exactly one PASS attempt with intact output hashes."""
    root = Path(root); rows = []; failed_attempts = 0; jobs = 0; elapsed = 0.
    wanted = None if plan is None else {j['id']: j for j in json.loads(Path(plan).read_text())['jobs']}
    names = sorted(p.name for p in (root/'jobs').iterdir() if p.is_dir())
    if wanted is not None and set(names) != set(wanted): raise ValueError('Run directory and plan list different jobs')
    for name in names:
        attempts = sorted(p for p in (root/'jobs'/name).glob('attempt_*') if p.is_dir()); good = []
        for a in attempts:
            try: m = runtime.verify_outputs(a); good.append((a, m))
            except (ValueError, OSError): failed_attempts += 1
        if len(good) != 1: raise ValueError(f'{name}: expected exactly one PASS attempt, found {len(good)}')
        a, m = good[0]
        if wanted is not None and m['config_hash'] != wanted[name]['config_hash']: raise ValueError(f'{name}: config differs from the plan')
        rr = json.loads((a/'rows.json').read_text())
        if any(q.get('status', 'OK') != 'OK' for q in rr): raise ValueError(f'{name}: non-OK rows')
        rows.extend(rr); jobs += 1; elapsed += m['elapsed_seconds']
    return rows, dict(run=str(root), jobs=jobs, rows=len(rows), failed_attempts_retained=failed_attempts, job_cpu_seconds=elapsed)


def cell_summary(rows):
    """Coverage and width of one (condition, method) cell; rows are one per seed."""
    n = len(rows); cov = sum(bool(q['covered']) for q in rows); empty = sum(bool(q.get('empty_confidence_set')) for q in rows)
    out = dict(n=n, seeds=len({q['seed'] for q in rows}), covered=cov, coverage=cov/n, wilson_95=wilson(cov, n), empty_confidence_sets=empty,
               coverage_empty_as_miss=(cov - empty)/n, wilson_95_empty_as_miss=wilson(cov - empty, n),
               mean_width=float(np.mean([q['width'] for q in rows])), width_sd=float(np.std([q['width'] for q in rows], ddof=1)) if n > 1 else None)
    if out['seeds'] != n: raise ValueError('A cell must hold one row per seed')
    return out


def paired_reduction(rows_ref, rows_cand):
    """Seed-paired relative width reduction 100(1 - W_cand/W_ref), averaged over seeds with a t interval."""
    ref = {q['seed']: q['width'] for q in rows_ref}; cand = {q['seed']: q['width'] for q in rows_cand}
    if set(ref) != set(cand): raise ValueError('Unpaired seeds')
    return mean_t([100*(1 - cand[k]/max(ref[k], 1e-12)) for k in sorted(ref)])
