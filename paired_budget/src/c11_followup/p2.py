"""Paired comparison of the learned false-positive budget with the true false-positive rate. Pure functions; no file input or output."""
from collections import defaultdict
import hashlib
import numpy as np
from c11_upgrade.population import periodic_law, lp_interval, law_limits
from c11_upgrade.inference import cp_upper, cp_interval
from c11_upgrade.addendum_core import (INFORMATION, rng, population_references,
    solve_information, interval_metrics, make_problem, validate_law)
from c11_upgrade.e9 import sample_words

BUDGETS = (200, 500, 2000)
ARMS = ('oracle_f_05', 'oracle_f_04', 'learned', 'plugin')
KNOWN_ZERO = (310068, 310080)
CONFIRM_SEEDS = tuple(range(480001, 480201))
FIXTURE_SEEDS = (990001, 990002, 990003)
BOOTSTRAP_SEED = 202610150803


def law_specs():
    return [dict(law_seed=310041+i, law_index=i, alphabet=4) for i in range(40)]


def approved_spec(spec):
    if spec not in law_specs():
        raise ValueError('Unregistered law specification')
    return dict(spec)


def sample_hash(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.asarray(array, dtype='<i8')
        h.update(str(a.shape).encode('ascii')); h.update(a.tobytes())
    return h.hexdigest()


def generate_samples(law, condition_id, seed, budget, n_obs=10000):
    _, oy = sample_words(rng(8, condition_id, seed, 0), law, n_obs)
    ax, ay = sample_words(rng(8, condition_id, seed, 1), law, budget)
    cx, cy = sample_words(rng(8, condition_id, seed, 2), law, budget)
    return oy, ax[:, 1], ay, cx[:, 1], cy[:, 1]


def run_replicate(spec, budget, seed, *, confirmatory=False, law=None, reference=None,
                  auxiliary=True):
    spec = approved_spec(spec)
    if budget not in BUDGETS or isinstance(budget, bool):
        raise ValueError('Unregistered budget')
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError('Integer replicate seed required')
    if confirmatory:
        if seed not in CONFIRM_SEEDS or law is not None:
            raise ValueError('Full mode requires registered confirmatory seeds and generated law')
        law = periodic_law(spec['law_seed'], 3, 4)
    elif seed not in FIXTURE_SEEDS or law is None:
        raise ValueError('Smoke requires explicit fixture law and registered fixture seed')
    law = validate_law(law)
    ref = population_references(law) if reference is None else reference
    cid = 3*spec['law_index'] + BUDGETS.index(budget)
    oy, ax1, ay, cx1, cy1 = generate_samples(law, cid, seed, budget)
    hashes = dict(observation=sample_hash(oy), audit=sample_hash(ax1, ay),
                  calibration=sample_hash(cx1, cy1))
    k = int(np.sum((cx1 == 0) & (cy1 == 1)))
    settings = {'oracle_f_05': (ref['f'], .05), 'oracle_f_04': (ref['f'], .04),
                'learned': (cp_upper(k, budget, .01), .04), 'plugin': (k/budget, .05)}
    if spec['law_seed'] in KNOWN_ZERO:
        if abs(ref['f']) > 1e-12:
            raise ValueError('known_zero fixture law must have f*=0')
        settings['known_zero'] = (0., .05)
    rows = []
    for arm, (epsilon, alpha) in settings.items():
        intervals = solve_information(oy, ax1, ay, epsilon, alpha)
        # Coverage always uses ref[f*]. Auxiliary epsilon-dependent sets never replace it.
        varying = ref['intervals'] if arm.startswith('oracle') or arm == 'known_zero' else (
            {i: lp_interval(make_problem(law_limits(law), epsilon, i)) for i in INFORMATION}
            if auxiliary else None)
        for info in INFORMATION:
            rows.append({**spec, 'condition_id': cid, 'budget': budget, 'seed': int(seed),
                'arm': arm, 'method': info, 'confirmatory': bool(confirmatory), 'n_obs': 10000,
                'audit_n': budget, 'calibration_n': budget, 'calibration_k': k,
                'epsilon': float(epsilon), 'moment_alpha': alpha, 'true_f': ref['f'],
                'sample_hashes': hashes.copy(), **interval_metrics(intervals[info], ref['intervals'][info], ref['theta']),
                'auxiliary_epsilon_population_interval': None if varying is None else varying[info]})
    return rows


def ratio_of_mean_widths(widths):
    """Last dimension M/N/F, replicate axis first; fallback widths already equal two."""
    a = np.asarray(widths, float)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 2+1e-8):
        return {'value': None, 'undefined_reason': 'invalid_widths'}
    means = a.mean(axis=0)
    if means[0] == 0:
        return {'value': None, 'undefined_reason': 'zero_denominator'}
    return {'value': float(1-means[1]/means[0]), 'undefined_reason': None}


def paired_resample(tensor, indices):
    """tensor = laws x replicates x arms x M/N/F; one index array per law."""
    a, indices = np.asarray(tensor), np.asarray(indices)
    if a.ndim != 4 or a.shape[2:] != (4, 3) or indices.shape != a.shape[:2]:
        raise ValueError('Expected paired law x replicate indices and four-arm M/N/F tensor')
    if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= a.shape[1]):
        raise ValueError('Invalid replicate index')
    return a[np.arange(len(a))[:, None], indices]


def law_contrasts(tensor):
    a = np.asarray(tensor, float)
    if a.ndim != 4 or a.shape[2:] != (4, 3) or a.shape[1] == 0:
        raise ValueError('Expected law x replicate x four arms x M/N/F')
    means = a.mean(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        g = 1-means[:, :, 1]/means[:, :, 0]
    g[means[:, :, 0] == 0] = np.nan
    return g, g[:, 0]-g[:, 2]


def paired_bootstrap(tensor, reps=2000, seed=BOOTSTRAP_SEED):
    a = np.asarray(tensor, float)
    if not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 2+1e-8):
        raise ValueError('Invalid bootstrap widths')
    g, delta = law_contrasts(a)
    defined = np.isfinite(g).all(axis=1) & np.isfinite(delta)
    point = float(np.median(delta[defined])) if defined.any() else None
    if not defined.all():
        return dict(point=point, defined_laws=int(defined.sum()), ci_95=None,
                    status='not_evaluable', reason='undefined_law_ratio', valid_bootstrap_reps=0)
    gen = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        idx = gen.integers(0, a.shape[1], size=a.shape[:2])
        g_star, d = law_contrasts(paired_resample(a, idx))
        # Never silently drop an undefined bootstrap draw or change its law panel.
        if not np.isfinite(g_star).all() or not np.isfinite(d).all():
            return dict(point=point, defined_laws=len(delta), ci_95=None,
                        status='not_evaluable', reason='undefined_bootstrap_ratio', valid_bootstrap_reps=len(values))
        values.append(float(np.median(d)))
    ci = np.quantile(values, [.025, .975], method='linear').tolist()
    return dict(point=point, defined_laws=len(delta), ci_95=ci, status='ok',
                valid_bootstrap_reps=reps, uncertainty='approximate_MC_percentile_fixed_law_panel')


def law_resampling_sensitivity(tensor, reps=2000, seed=BOOTSTRAP_SEED+1):
    _, delta = law_contrasts(tensor)
    if not np.isfinite(delta).all():
        return {'status': 'not_evaluable', 'reason': 'undefined_law_ratio'}
    gen = np.random.default_rng(seed)
    values = np.median(delta[gen.integers(0, len(delta), size=(reps, len(delta)))], axis=1)
    return {'status': 'descriptive_law_resampling_sensitivity',
            'ci_95': np.quantile(values, [.025, .975], method='linear').tolist()}


def complete_grid(rows):
    """Reject duplicates, missing cells, altered sample contracts, and unpaired samples."""
    expected = {(s['law_index'], b, seed, a, i) for s in law_specs() for b in BUDGETS
        for seed in CONFIRM_SEEDS for a in (*ARMS, *(('known_zero',) if s['law_seed'] in KNOWN_ZERO else ()))
        for i in INFORMATION}
    seen, shared = set(), {}
    for r in rows:
        try:
            key = (r['law_index'], r['budget'], r['seed'], r['arm'], r['method'])
            spec = law_specs()[r['law_index']]
            if key not in expected or key in seen or any(r.get(k) != v for k, v in spec.items()):
                return False
            if (r.get('confirmatory') is not True or r.get('n_obs') != 10000 or
                r.get('audit_n') != r['budget'] or r.get('calibration_n') != r['budget'] or
                r.get('condition_id') != 3*r['law_index']+BUDGETS.index(r['budget'])):
                return False
            h = r['sample_hashes']
            if set(h) != {'observation', 'audit', 'calibration'} or any(not isinstance(v, str) or len(v) != 64 for v in h.values()):
                return False
            if key[:3] in shared and shared[key[:3]] != h:
                return False
            shared[key[:3]] = h
            seen.add(key)
        except (KeyError, TypeError, ValueError, IndexError):
            return False
    return seen == expected


def summarize(rows, *, confirmatory=False, bootstrap_reps=2000, bootstrap_seed=BOOTSTRAP_SEED):
    rows = list(rows)
    full = complete_grid(rows) if confirmatory else False
    groups = defaultdict(list)
    for r in rows:
        groups[r['law_index'], r['budget'], r['arm'], r['method']].append(r)
    output = []
    for key, group in sorted(groups.items()):
        n = len(group); defined = [r for r in group if r['population_reference_defined']]
        hits = sum(r['contains_population'] is True for r in defined)
        output.append(dict(law_index=key[0], budget=key[1], arm=key[2], method=key[3], n=n,
            reference_defined_n=len(defined), reference_undefined_n=n-len(defined),
            contains_population_hits=hits, contains_population=hits/len(defined) if defined else None,
            contains_population_total_denominator=hits/n,
            coverage_mc_95=list(cp_interval(hits,len(defined))) if defined else None,
            covers_truth=sum(r['covers_truth'] is True for r in group)/n,
            empty_n=sum(r['empty_feasible'] for r in group),
            solver_failure_n=sum(r['solver_failure'] for r in group),
            fallback_n=sum(r['fallback_used'] for r in group),
            fallback_population_known_n=sum(r['fallback_contains_population'] is not None for r in group),
            fallback_population_hits=sum(r['fallback_contains_population'] is True for r in group),
            mean_width=float(np.mean([r['width'] for r in group]))))
    result = dict(full_grid_complete=full, row_count=len(rows), groups=output,
                  h1='not_evaluable' if confirmatory else 'fixture_not_confirmatory')
    if not full:
        result['reason'] = 'incomplete_or_invalid_grid' if confirmatory else 'fixture_only'
        return result
    tensors, comparisons = {}, []
    for b in BUDGETS:
        a = np.empty((40, 200, 4, 3))
        for j in range(40):
            for arm_idx, arm in enumerate(ARMS):
                for info_idx, info in enumerate(INFORMATION):
                    a[j, :, arm_idx, info_idx] = [r['width'] for r in sorted(groups[j,b,arm,info], key=lambda r:r['seed'])]
        tensors[b] = a
        g, delta = law_contrasts(a)
        for j in range(40):
            comparisons.append(dict(law_index=j, budget=b,
                g={arm:float(g[j,k]) if np.isfinite(g[j,k]) else None for k,arm in enumerate(ARMS)},
                delta=float(delta[j]) if np.isfinite(delta[j]) else None,
                oracle05_minus_oracle04=float(g[j,0]-g[j,1]) if np.isfinite(g[j,:2]).all() else None,
                oracle04_minus_learned=float(g[j,1]-g[j,2]) if np.isfinite(g[j,1:3]).all() else None))
    primary = paired_bootstrap(tensors[500], bootstrap_reps, bootstrap_seed)
    result.update(comparisons=comparisons, primary=primary,
        law_panel_sensitivity=law_resampling_sensitivity(tensors[500], bootstrap_reps, bootstrap_seed+1),
        h2_descriptive={'bank_A_stored_reference': .2307, 'not_causal_decomposition': True,
           'bank_C_oracle05_median': float(np.nanmedian(law_contrasts(tensors[500])[0][:,0]))
             if np.isfinite(law_contrasts(tensors[500])[0][:,0]).any() else None})
    references_defined = all(r['population_reference_defined'] for r in rows)
    if primary['status'] == 'ok' and references_defined:
        result['h1'] = 'prediction_consistent' if primary['point'] >= .05 and primary['ci_95'][0] > 0 else 'prediction_not_met'
    else:
        result['reason'] = 'undefined_reference_or_ratio'
    return result
