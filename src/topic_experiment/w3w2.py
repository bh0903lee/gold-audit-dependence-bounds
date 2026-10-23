"""Runners for the prespecified six-variable comparison ('w3') and the assumption-violation study ('w2a', 'w2b').

Selected by the config key 'experiment'; the designs are described in the manuscript sections on the six-variable
comparison and on assumption violations.  One job = one seed.  Sample arrays are not stored; the
count tables saved in counts.npz are sufficient statistics for every interval computed here.
"""
import numpy as np
from . import engine, sixvar, violations
from .common import AuditRows

L = 3
GUARANTEE = {'word': True, 'sixvar': True, 'l1ball': True, 'bootstrap': False, 'dirichlet': False}
METHOD_TAG = {'bootstrap': 4, 'dirichlet': 5}
W2_STREAM_TAG = 2


def interval_for(method, y, ax, ay, alpha, seed_key):
    """(low, high, info) for one method name; engine.InfeasibleConfidenceSet means an empty confidence set."""
    family, _, interval = method.partition('/')
    if family == 'word':
        lo, hi = engine.bounds(y, ax, ay, alpha=alpha, moments='neighbor', interval=interval); return lo, hi, {}
    if family == 'sixvar':
        lo, hi = sixvar.six_variable_bounds(y, ax, ay, alpha=alpha, interval=interval); return lo, hi, {}
    oc = sixvar.obs_counts(y); ac = sixvar.audit_counts5(ax, ay)
    if family == 'l1ball':
        lo, hi = sixvar.l1_ball_bounds(oc, ac, alpha); return lo, hi, {}
    rng = np.random.default_rng(list(seed_key) + [METHOD_TAG[family]])
    if family == 'bootstrap':
        return sixvar.bootstrap_interval(oc, ac, alpha, rng)
    if family == 'dirichlet':
        return sixvar.dirichlet_interval(oc, ac, alpha, rng)
    raise ValueError('Unknown method ' + method)


def evaluate(method, y, ax, ay, alpha, seed_key, truth, pop=None):
    """One result row fragment; empty confidence sets are retained as flagged [-1,1] fallbacks.

    Solver failures (engine.SolverFailure) and other errors propagate, so they are never counted as empty sets."""
    try:
        lo, hi, info = interval_for(method, y, ax, ay, alpha, seed_key)
    except engine.InfeasibleConfidenceSet as error:
        row = dict(empty_confidence_set=True, error=str(error), low=-1., high=1., width=2., covered=True)
        if pop is not None: row['contains_population'] = True
        return row
    row = dict(empty_confidence_set=False, low=float(lo), high=float(hi), width=float(max(hi - lo, 0.)), covered=bool(lo <= truth <= hi), **info)
    if pop is not None:
        row['contains_population'] = bool(lo <= pop[0] + 1e-9 and hi >= pop[1] - 1e-9)
    return row


def run_w3(c, out):
    rows = AuditRows(out); detect = violations.DETECT; alpha = c['alpha']; counts = {}
    for seed in c['seeds']:
        rng = np.random.default_rng(seed)
        for stay in c['stay']:
            law = violations.analyst_law(violations.make_condition('baseline', 0, stay))
            pop = [float(v) for v in sixvar.closed_form(*[law[k] for k in sixvar.NAMES])]; truth = 2*stay - 1
            _, y = engine.sample_episodes(rng, c['episodes'], L, stay, detect)
            for budget in c['audit_budgets']:
                x, ay = engine.sample_episodes(rng, budget, L, stay, detect); ax = x[:, engine.AUDIT_T]
                counts[f'obs_{seed}_{stay}'] = sixvar.obs_counts(y); counts[f'aud_{seed}_{stay}_{budget}'] = sixvar.audit_counts16(ax, ay)
                for method in c['methods']:
                    rows.append(dict(experiment='w3', seed=seed, stay=stay, L=L, audits=budget, method=method, guarantee=GUARANTEE[method.split('/')[0]],
                                     truth=truth, population_low=pop[0], population_high=pop[1], status='OK',
                                     **evaluate(method, y, ax, ay, alpha, [seed, int(round(1000*stay)), budget], truth, pop)))
    np.savez_compressed(out/'counts.npz', **counts)
    return rows


def run_w2(c, out):
    rows = AuditRows(out); alpha = c['alpha']; counts = {}
    for seed in c['seeds']:
        for si, stay in enumerate(c['stay']):
            for axis, level in violations.grid(c['axes']):
                cond = violations.make_condition(axis, level, stay); truth = violations.analyst_law(cond)['truth']
                # Common random numbers across conditions. The constant tag keeps this stream distinct from the six-variable
                # comparison's default_rng(seed): numpy zero-pads seed sequences, so [seed, 0] would equal that stream.
                rng = np.random.default_rng([seed, si, W2_STREAM_TAG])
                y, audits = violations.draw_condition(rng, cond, c['episodes'], c['audit_budgets'], L)
                tag = f'{seed}_{stay}_{axis}_{level}'; counts['obs_' + tag] = sixvar.obs_counts(y)
                for budget in c['audit_budgets']:
                    ax, ay, _ = audits[budget]; counts[f'aud_{tag}_{budget}'] = sixvar.audit_counts16(ax, ay)
                    t1 = int(np.sum((ax == 0) & (ay[:, engine.AUDIT_T] == 1)))
                    for method in c['methods']:
                        rows.append(dict(experiment='w2', seed=seed, stay=stay, L=L, axis=axis, level=str(level), audits=budget, method=method,
                                         truth=truth, t1_count=t1, status='OK',
                                         **evaluate(method, y, ax, ay, alpha, [seed, int(round(1000*stay)), budget], truth)))
    np.savez_compressed(out/'counts.npz', **counts)
    return rows


def run(c, out):
    if c.get('lengths', [L]) != [L]:
        raise ValueError('The six-variable comparison and the violation study use three-point windows only (lengths must be [3])')
    if c['experiment'] == 'w3':
        return run_w3(c, out)
    if c['experiment'] in ('w2a', 'w2b'):
        return run_w2(c, out)
    raise ValueError('Unknown experiment')


def job_summary(rows):
    """Per-job descriptive summary only; the cross-seed analysis is in manuscript/supplement/sixvar_violations/."""
    cells = {}
    for r in rows:
        key = '|'.join(str(r.get(k, '')) for k in ('axis', 'level', 'stay', 'audits', 'method'))
        g = cells.setdefault(key, dict(rows=0, covered=0, empty_confidence_sets=0, width_sum=0.))
        g['rows'] += 1; g['covered'] += bool(r['covered']); g['empty_confidence_sets'] += bool(r.get('empty_confidence_set')); g['width_sum'] += r['width']
    return dict(topic='C11', experiment=rows[0]['experiment'], kind='per_job_summary', cells=cells,
                seeds=sorted({r['seed'] for r in rows}),
                warning='One seed per job: no interval, coverage or claim is derived here. See manuscript/supplement/sixvar_violations/analyze_sixvar.py and analyze_violations.py.')
