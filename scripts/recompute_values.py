"""Value check of the stored results: file integrity, recomputed value files and the inventory of every reported number.

1. Integrity. Every file in manuscript/supplement/ must be listed in manuscript/supplement/supplement_manifest.json with its
   SHA-256 digest, and every listed file must exist (missing, extra or changed files fail). The SPHERE files in data/raw/ must
   match the digests stored in extensions/sphere_summary.json, and the UCI archive the digest in room_occupancy_manifest.json.
2. Recomputed value files (inputs: main_study_rows.csv, main_study_slack_attribution.json, main_study_independent_replay.json,
   sixvar_violations/analysis_*.json, extensions/*_summary.json, extensions/fp_gap_certificate.json):
     main_study_summary.json, population_summary.json, main_study_full_audit_fallbacks.csv   (simulation study)
     sixvar_violation_numbers.json                                                         (six-variable comparison, violation study)
     extension_numbers.json                                                                (extension studies)
   Each file is first compared byte for byte; if the bytes differ, numbers are compared with relative tolerance 1e-9
   (absolute 1e-12) and all other entries exactly.
3. Reported numbers. reported_values.json lists every number printed in the article and its supplementary tables that comes
   from the stored results, as printed, with an expression that reads it from the stored files (J: JSON file, C: CSV file,
   L: JSON-lines file, find: the unique record with the given fields, frac: a stored fraction). Each expression is evaluated
   and must round to the printed value (or satisfy the stated bound, for numbers printed as 'below', 'at most' or 'at least').

Exit code 0 if every check passes, 1 otherwise. No experiment is run and nothing is written unless --write DIR is given (then
the recomputed value files of step 2 are written to DIR instead of compared).

  python scripts/recompute_values.py
  python scripts/recompute_values.py --supplement DIR   # check another copy of the supplement folder
"""
import argparse, ast, csv, hashlib, io, itertools, json, math, statistics, sys
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
REL_TOL, ABS_TOL = 1e-9, 1e-12


# ---------------------------------------------------------------- main simulation study
def main_study(sup):
    rows = list(csv.DictReader((sup / 'main_study_rows.csv').open()))
    for r in rows:
        for k in ['stay', 'low', 'high', 'width', 'population_low', 'population_high', 'population_width', 'concentration_slack']: r[k] = float(r[k]) if r[k] else None
        for k in ['seed', 'L', 'audits']: r[k] = int(r[k])
        for k in ['covered', 'empty_confidence_set']: r[k] = r[k] == 'True'
    slack = json.loads((sup / 'main_study_slack_attribution.json').read_text())['rows']
    replay = json.loads((sup / 'main_study_independent_replay.json').read_text())['rows']
    freeze_date = json.loads((sup / 'main_study_input_manifest.json').read_text())['freeze_date']

    def select(source='both_sampled', **kw): return [r for r in rows if r['source'] == source and all(r[k] == v for k, v in kw.items())]

    def stats(a):
        a = np.asarray(a, float); mean = float(a.mean()); se = float(a.std(ddof=1) / np.sqrt(len(a))); d = float(t.ppf(.975, len(a) - 1)) * se
        return dict(mean=mean, se=se, lo=mean - d, hi=mean + d, n=len(a))

    def paired(cand, ref):
        """Mean seedwise relative reduction 1-W_cand/W_ref with Student-t Monte Carlo interval."""
        a = {r['seed']: r['width'] for r in cand}; b = {r['seed']: r['width'] for r in ref}; assert a.keys() == b.keys() and len(a) == 200
        return stats([1 - a[s] / b[s] for s in sorted(a)])

    def population(stay):
        """Population moments and identified intervals from the stated generative law."""
        words = np.array(list(itertools.product(range(3), repeat=3)))
        x = (words > 0).astype(float); y = (words == 2).astype(float)
        prob = np.zeros(len(words))
        for j, word in enumerate(words):
            for prev in [0, 1]:
                mass = .5; last = prev
                for z in word:
                    xx = int(z > 0); yy = int(z == 2)
                    mass *= stay if xx == last else 1 - stay
                    if xx: mass *= ([.95, .2][last] if yy else 1 - [.95, .2][last])
                    last = xx
                prob[j] += mass
        features = np.array([x[:, 1], y[:, 1], y[:, 0] * x[:, 1], x[:, 1] * y[:, 2], y[:, 0] * y[:, 1]])
        m, yy, a, b, h = features @ prob
        neighbor = [max(1 - 4 * m + 4 * a + 4 * b - 4 * h, 4 * m - 3), 1 - 4 * yy + 4 * min(a, b)]
        eq = np.array([np.ones(len(words))] + [np.all(words[:, :2] == u, axis=1).astype(float) - np.all(words[:, 1:] == u, axis=1) for u in itertools.product(range(3), repeat=2)]
                      + [np.all(y == u, axis=1).astype(float) for u in itertools.product(range(2), repeat=3)] + [x[:, 1]])
        objective = (2 * x[:, 0] - 1) * (2 * x[:, 1] - 1)
        fits = [linprog(sign * objective, A_eq=eq, b_eq=eq @ prob, bounds=(0, None), method='highs') for sign in [1, -1]]
        assert all(f.success for f in fits)
        return dict(stay=stay, m=m, y=yy, a=a, b=b, h=h, neighbor=neighbor, marginal=[float(objective @ f.x) for f in fits], width=neighbor[1] - neighbor[0], true_theta=2 * stay - 1)

    populations = [population(s) for s in [.2, .5, .8]]
    summary = {'freeze_date': freeze_date, 'collection': 'v4_2', 'rows': len(rows), 'primary': [], 'original': [], 'full_vs_neighbor': [], 'coverage': [], 'containment': {}, 'slack': [],
               'empty_sets': [r for r in rows if r['empty_confidence_set']], 'replay_consistency': {}, 'grid': []}

    def contains(r): return r['low'] <= r['population_low'] + 1e-9 and r['high'] >= r['population_high'] - 1e-9
    assert all(r['status'] == 'OK' for r in rows)
    for stay in [.2, .5, .8]:
        kw = dict(stay=stay, L=3, audits=500, moments='neighbor')
        h = select(**kw, interval='hoeffding'); eb = select(**kw, interval='empirical_bernstein'); cp = select(**kw, interval='exact_binomial')
        summary['primary'].append(dict(stay=stay, hoeffding=stats([r['width'] for r in h]), eb=stats([r['width'] for r in eb]), cp=stats([r['width'] for r in cp]), eb_vs_h=paired(eb, h), cp_vs_h=paired(cp, h), cp_vs_eb=paired(cp, eb),
                                       covered=dict(hoeffding=sum(r['covered'] for r in h), eb=sum(r['covered'] for r in eb), cp=sum(r['covered'] for r in cp))))
        ma = select(stay=stay, L=3, audits=500, moments='marginal', interval='hoeffding'); z = paired(h, ma)
        summary['original'].append(dict(stay=stay, marginal=stats([r['width'] for r in ma]), neighbor=stats([r['width'] for r in h]), reduction=z, meets_20pct=bool(z['mean'] >= .2)))
        for L in [3, 4]:
            for interval in ['empirical_bernstein', 'exact_binomial']:
                fu = select(stay=stay, L=L, audits=500, moments='full', interval=interval); nb = select(stay=stay, L=L, audits=500, moments='neighbor', interval=interval)
                summary['full_vs_neighbor'].append(dict(stay=stay, L=L, interval=interval, reduction_including_fallbacks=paired(fu, nb), fallbacks=sum(r['empty_confidence_set'] for r in fu)))
        # Slack: fixed-allocation Shapley attribution (CP) from the archived arrays, and the native redistributed-allocation ratio per interval.
        s = [r for r in slack if r['stay'] == stay]
        native = {}
        for interval in ['hoeffding', 'empirical_bernstein', 'exact_binomial']:
            b = {r['seed']: r for r in select(**kw, interval=interval)}; eo = {r['seed']: r for r in select('exact_observations', **kw, interval=interval)}; ea = {r['seed']: r for r in select('exact_audit', **kw, interval=interval)}
            native[interval] = dict(audit_slack_fraction=stats([(eo[k]['width'] - b[k]['population_width']) / (b[k]['width'] - b[k]['population_width']) for k in b]),
                                    observation_slack_fraction=stats([(ea[k]['width'] - b[k]['population_width']) / (b[k]['width'] - b[k]['population_width']) for k in b]))
        summary['slack'].append(dict(stay=stay, cp_fixed_allocation_audit_fraction=stats([r['audit_fraction'] for r in s]), cp_fixed_allocation_total_slack=stats([r['total_slack'] for r in s]), native_redistributed_allocation=native))
    for L in [3, 4]:
        for method in ['marginal', 'neighbor', 'full']:
            for interval in ['hoeffding', 'empirical_bernstein', 'exact_binomial']:
                rr = select(L=L, moments=method, interval=interval)
                summary['coverage'].append(dict(L=L, moments=method, interval=interval, total=len(rr), empty_confidence_sets=sum(r['empty_confidence_set'] for r in rr), covered_with_fallback=sum(r['covered'] for r in rr),
                                                covered_counting_empty_as_miss=sum(r['covered'] and not r['empty_confidence_set'] for r in rr), population_interval_contained=sum(contains(r) for r in rr if not r['empty_confidence_set'])))
                for stay in [.2, .5, .8]:
                    for B in [50, 100, 500, 1000, 5000]:
                        rr = select(L=L, moments=method, interval=interval, stay=stay, audits=B); ok = [r for r in rr if not r['empty_confidence_set']]
                        summary['grid'].append(dict(L=L, moments=method, interval=interval, stay=stay, audits=B, mean_width_with_fallback=float(np.mean([r['width'] for r in rr])),
                                                    mean_width_success_only=float(np.mean([r['width'] for r in ok])), empty=len(rr) - len(ok), covered=sum(r['covered'] for r in rr)))
    both = select(); ok = [r for r in both if not r['empty_confidence_set']]
    summary['containment'] = dict(both_sampled_rows=len(both), returned=len(ok), population_interval_contained=sum(contains(r) for r in ok), true_target_covered=sum(r['covered'] for r in both),
                                  exceptions=[dict(seed=r['seed'], stay=r['stay'], L=r['L'], audits=r['audits'], moments=r['moments'], interval=r['interval'], low=r['low'], high=r['high'],
                                                   population_low=r['population_low'], population_high=r['population_high'], covered=r['covered']) for r in ok if not contains(r)])
    # Consistency with the saved-sample replay (same arrays, independent script); the maximum absolute differences are recorded.
    rd = {(r['seed'], r['stay']): r for r in replay}
    summary['replay_consistency'] = dict(
        cases=len(replay),
        max_abs_eb_width_difference=max(abs(r['width'] - rd[r['seed'], r['stay']]['corrected_eb_width']) for r in select(L=3, audits=500, moments='neighbor', interval='empirical_bernstein')),
        max_abs_cp_width_difference=max(abs(r['width'] - rd[r['seed'], r['stay']]['cp_width']) for r in select(L=3, audits=500, moments='neighbor', interval='exact_binomial')),
        max_abs_audit_fraction_difference=max(abs(r['audit_fraction'] - rd[r['seed'], r['stay']]['cp_fixed_alpha_audit_fraction']) for r in slack))
    # Full-audit/CP fallback report: each empty confidence set is recorded as a flagged [-1,1] row (width 2).
    fallback = []
    for L in [3, 4]:
        for stay in [.2, .5, .8]:
            for B in [50, 100, 500, 1000, 5000]:
                rr = select(L=L, stay=stay, audits=B, moments='full', interval='exact_binomial'); ok = [r for r in rr if not r['empty_confidence_set']]
                fallback.append(dict(L=L, stay=stay, audits=B, requested=len(rr), returned=len(ok), empty=len(rr) - len(ok), width_mean_success_only=float(np.mean([r['width'] for r in ok])),
                                     width_mean_with_fallback=float(np.mean([r['width'] for r in rr])), target_covered_success=sum(r['covered'] for r in ok), target_covered_with_fallback=sum(r['covered'] for r in rr)))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(fallback[0])); writer.writeheader(); writer.writerows(fallback)
    return {'main_study_summary.json': json.dumps(summary, indent=2) + '\n',
            'population_summary.json': json.dumps(populations, indent=2) + '\n',
            'main_study_full_audit_fallbacks.csv': out.getvalue()}


# ---------------------------------------------------------------- six-variable comparison and violation study
def sixvar_violations(sup):
    w3 = json.loads((sup / 'sixvar_violations/analysis_sixvar.json').read_text()); w2 = json.loads((sup / 'sixvar_violations/analysis_violations.json').read_text())
    STAYS = (0.2, 0.5, 0.8); WORD = 'word/exact_binomial'; SIX = 'sixvar/exact_binomial'
    conf = {(c['stay'], c['audits'], c['method']): c for c in w3['confirm']['cells']}; ext = {(c['stay'], c['audits'], c['method']): c for c in w3['extended']['cells']}
    pair = {(p['stay'], p['audits'], p['reference'], p['candidate']): p for p in w3['confirm']['paired_reductions']}
    cell = {(c['axis'], c['level'], c['stay'], c['audits'], c['method']): c for c in w2['set_a']['cells']}
    b = {(c['axis'], c['level'], c['stay'], c['audits'], c['method']): c for c in w2['set_b']['cells']}
    curves = {(c['curve'], c['stay']): c for c in w2['breakdown_curves']}
    small = (('V1', '0.002'), ('V1', '0.005'), ('V1', '0.01'), ('V1', '0.02'), ('V4a', '0.01'), ('V4a', '0.02'), ('V4a', '0.05'), ('V2a', 'stay+0.05'), ('V2a', 'det+0.05'), ('V2a', 'det+0.1'))
    numbers = dict(
        sixvar_reduction_B500={str(s): pair[(s, 500, WORD, SIX)] for s in STAYS},
        sixvar_reduction_reanalysis_B500={str(p['stay']): p for p in w3['reanalysis_v42']['paired_reductions'] if p['audits'] == 500 and p['reference'] == WORD and p['candidate'] == SIX},
        sixvar_other_intervals_B500={m: {str(s): pair[(s, 500, f'word/{m}', f'sixvar/{m}')]['mean'] for s in STAYS} for m in ('hoeffding', 'empirical_bernstein') if (0.2, 500, f'word/{m}', f'sixvar/{m}') in pair},
        verdict=w3['confirm']['verdict'], guaranteed_min_coverage_1000=min(c['coverage'] for c in w3['extended']['cells'] if c['guarantee']),
        guaranteed_min_containment_1000=min(c['contains_population'] for c in w3['extended']['cells'] if c['guarantee']),
        word_cp_empty_1000=sum(c['empty_confidence_sets'] for c in w3['extended']['cells'] if c['method'] == WORD),
        comparator_validity=w3['extended']['comparator_validity'],
        comparator_B50_confirm={m: {str(s): conf[(s, 50, m)]['coverage'] for s in STAYS} for m in ('bootstrap', 'dirichlet')},
        comparator_reduction_range_B500={m: [min(pair[(s, 500, WORD, m)]['mean'] for s in STAYS), max(pair[(s, 500, WORD, m)]['mean'] for s in STAYS)] for m in ('bootstrap', 'dirichlet', 'l1ball')},
        comparator_containment_range_1000={m: [min(ext[(s, B, m)]['contains_population'] for s in STAYS for B in (100, 500)), max(ext[(s, B, m)]['contains_population'] for s in STAYS for B in (100, 500))] for m in ('bootstrap', 'dirichlet')},
        breakdown_first_positive={k[0] + '@' + str(k[1]): dict(first_d_positive=v['first_level_with_d_positive'], first_incompatible=v['first_incompatible_level']) for k, v in curves.items()},
        t1={f'{ax}:{lv}:{s}:{B}': dict(rate=cell[(ax, lv, s, B, WORD)]['t1_rejection_rate'], analytic=cell[(ax, lv, s, B, WORD)]['t1_power_analytic'], pass_n=cell[(ax, lv, s, B, WORD)]['t1_pass_n'], pass_coverage=cell[(ax, lv, s, B, WORD)]['t1_pass_coverage'])
            for ax, lv in (('V1', '0.002'), ('V1', '0.005'), ('V1', '0.01'), ('V1', '0.02'), ('V1', '0.05'), ('V4a', '0.01'), ('V5', '2'), ('V2a', 'stay+0.2'), ('V4b', '0.2')) for s in STAYS for B in (500, 5000)},
        sixvar_under_violation={f'{ax}:{lv}:{s}:{B}': dict(coverage=cell[(ax, lv, s, B, SIX)]['coverage'], empty=cell[(ax, lv, s, B, SIX)]['empty_confidence_sets'], word_coverage=cell[(ax, lv, s, B, WORD)]['coverage'], word_empty=cell[(ax, lv, s, B, WORD)]['empty_confidence_sets'])
                                for ax, lv, s, B in (('V1', '0.1', 0.8, 500), ('V5', '2', 0.2, 500), ('V5', '2', 0.2, 5000))},
        min_coverage_small_violations=min(cell[(ax, lv, s, B, WORD)]['coverage_empty_as_miss'] for ax, lv in small for s in STAYS for B in (500, 5000)),
        min_coverage_anchoring=min(cell[('V4b', lv, s, B, WORD)]['coverage_empty_as_miss'] for lv in ('0.05', '0.1', '0.2') for s in STAYS for B in (500, 5000)),
        set_b=dict(V3_min_coverage=min(c['coverage'] for k, c in b.items() if k[0] in ('V3a', 'V3b') and k[4] in (WORD, SIX)), V3_cells=sum(1 for k in b if k[0] in ('V3a', 'V3b') and k[4] == WORD),
                   V6_word_min_coverage=min(c['coverage'] for k, c in b.items() if k[0] == 'V6' and k[4].startswith('word')), V6_six_min_coverage=min(c['coverage'] for k, c in b.items() if k[0] == 'V6' and k[4] == SIX),
                   V2b_word_empty_fraction={f'{lv}:{s}': b[('V2b', lv, s, 500, WORD)]['empty_confidence_sets'] / b[('V2b', lv, s, 500, WORD)]['n'] for lv in ('0.05', '0.1', '0.2') for s in STAYS},
                   V2b_six_min_coverage=min(c['coverage'] for k, c in b.items() if k[0] == 'V2b' and k[4] == SIX), V2b_six_empty=sum(c['empty_confidence_sets'] for k, c in b.items() if k[0] == 'V2b' and k[4] == SIX),
                   V2b_six_width_s02_B500=[b[('V2b', lv, 0.2, 500, SIX)]['mean_width'] for lv in ('0.05', '0.1', '0.2')]),
        runs=dict(w3=w3['run'], w2a=w2['set_a']['run'], w2b=w2['set_b']['run']))
    return {'sixvar_violation_numbers.json': json.dumps(numbers, indent=1) + '\n'}


# ---------------------------------------------------------------- extension studies
def extensions(sup):
    ext = sup / 'extensions'
    J = lambda n: json.loads((ext / n).read_text())
    d1, d2, e8, e9, e7, e7h, cert = (J(n) for n in ('fp_population_summary.json', 'fp_budget_summary.json', 'overlap_summary.json', 'laws_summary.json', 'application_summary.json', 'application_thinning_summary.json', 'fp_gap_certificate.json'))
    NUM = {}
    GRID = [(s, p) for s in (.2, .5, .8) for p in (0, .02, .10)]
    grp = {(g['condition_id'], g['method']): g for g in d2['groups'] if g['family'] == 'primary'}
    rng = lambda vals: [min(vals), max(vals)]
    dropped = dict(learned_coverage=rng([grp[(c, 'learned_upper')]['contains_population'] for c in range(9)]), learned_mc_interval=rng([x for c in range(9) for x in grp[(c, 'learned_upper')]['mc_ci']]),
                   plugin_coverage=rng([grp[(c, 'plugin_joint')]['contains_population'] for c in range(9)]))
    NUM['d2'] = dict(decision=d2['coverage_decision'], practical=d2['practical_precision'], replicates=d2['replicates'], dropped_columns=dropped,
                     cells={f'{s},{p}': dict(learned=grp[(c, 'learned_upper')]['contains_population'], naive=grp[(c, 'naive_zero')]['contains_population']) for c, (s, p) in enumerate(GRID)})
    NUM['d1'] = dict(counts=d1['counts'], h1=d1['h1'], witness=dict(seed=cert['seed'], epsilon=cert['epsilon'], gaps=cert['gaps'], verified=cert['verified'], N6=[c['objective'] for c in cert['certificates']['N6']], F=[c['objective'] for c in cert['certificates']['F']]))
    g8 = {(g['condition_id'], g['method']): g for g in e8['groups']}
    NUM['e8'] = dict(h8a=e8['h8a'], h8b=e8['h8b'], selection=e8['selection']['method'], min_naive_coverage=min(g8[(c, 'naive_overlap')]['contains_population'] for c in range(7)))
    c500 = [x for x in e9['comparisons'] if x['budget'] == 500]
    NUM['e9'] = dict(h9b=e9['h9b'], success=e9['h9b_success_laws'], total=e9['h9b_total_laws'], A_narrower=sum(x['N_vs_M']['strictly_narrower'] for x in c500 if x['bank'] == 'A'),
                     C_narrower=sum(x['N_vs_M']['strictly_narrower'] for x in c500 if x['bank'] == 'C'), h9a_below=e9['h9a_below_threshold'])
    L1 = e7['lags'][0]; ref = L1['reference']; G = {(g['budget'], g['info']): g for g in L1['groups']}
    NUM['e7'] = dict(frame=L1['frame_size'], unit_days=L1['unit_days'], theta=ref['theta'], f=ref['f'], fn=ref['false_negative_rate'], residual=ref['stationarity_residual'], intervals=ref['intervals'],
                     width_B200={i: G[(200, i)]['mean_width'] for i in 'MNF'}, width_B50={i: G[(50, i)]['mean_width'] for i in 'MNF'}, containment_B200={i: G[(200, i)]['contains_population'] for i in 'MNF'},
                     criterion=L1['application_criterion'], handoff=dict(method=e7h['selection']['method'], frame=e7h['lags'][0]['frame_size'], criterion=e7h['lags'][0]['application_criterion']), gate=L1['gate']['status'])
    return {'extension_numbers.json': json.dumps(NUM, indent=1)}


# ---------------------------------------------------------------- integrity
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def supplement_files(sup):
    return {p.relative_to(sup).as_posix() for p in sup.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'supplement_manifest.json'}


def integrity(sup, data):
    """Yield one message per missing, extra or changed file."""
    listed = {r['path']: r['sha256'] for r in json.loads((sup / 'supplement_manifest.json').read_text())['files']}
    present = supplement_files(sup)
    yield from (f'missing file   manuscript/supplement/{n}' for n in sorted(listed.keys() - present))
    yield from (f'extra file     manuscript/supplement/{n}' for n in sorted(present - listed.keys()))
    yield from (f'changed file   manuscript/supplement/{n}' for n in sorted(listed.keys() & present) if sha256(sup / n) != listed[n])
    raw = {n: d for n, d in json.loads((sup / 'extensions/sphere_summary.json').read_text())['input_sha256'].items()}
    on_disk = {p.relative_to(data / 'sphere_house/raw').as_posix() for p in (data / 'sphere_house/raw').rglob('*') if p.is_file()}
    yield from (f'SPHERE input missing or extra: {n}' for n in sorted(raw.keys() ^ on_disk))
    yield from (f'SPHERE input changed: {n}' for n in sorted(raw.keys() & on_disk) if sha256(data / 'sphere_house/raw' / n) != raw[n])
    uci = json.loads((sup / 'extensions/room_occupancy_manifest.json').read_text())['source_files']['room_occupancy_estimation.zip']
    if sha256(data / 'room_occupancy_estimation/room_occupancy_estimation.zip') != uci: yield 'UCI archive changed'


# ---------------------------------------------------------------- reported numbers
ALLOWED = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Store, ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.BinOp, ast.UnaryOp,
           ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.comprehension, ast.operator, ast.unaryop, ast.boolop, ast.cmpop)


def evaluator(sup):
    """Return a function that evaluates an inventory expression: arithmetic, indexing, comprehensions and the functions below only."""
    cache = {}

    def J(name):
        if name not in cache: cache[name] = json.loads((sup / name).read_text())
        return cache[name]

    def C(name):
        if name not in cache: cache[name] = parse(name, (sup / name).read_text())
        return cache[name]

    def L(name):
        if name not in cache: cache[name] = [json.loads(line) for line in (sup / name).read_text().splitlines() if line.strip()]
        return cache[name]

    def find(records, *pairs):
        hits = [r for r in records if all(r[k] == v for k, v in zip(pairs[::2], pairs[1::2]))]
        if len(hits) != 1: raise KeyError(f'{len(hits)} records match {pairs}')
        return hits[0]

    funcs = dict(J=J, C=C, L=L, find=find, frac=lambda v: float(Fraction(v)), min=min, max=max, sum=sum, len=len, abs=abs, sorted=sorted, all=all, any=any, median=statistics.median, float=float, int=int, range=range, zip=zip)

    def evaluate(expr):
        tree = ast.parse(expr, mode='eval')
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED) or isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in funcs and not node.keywords):
                raise ValueError(f'construct not allowed: {type(node).__name__}')
        return eval(compile(tree, '<inventory>', 'eval'), {'__builtins__': {}, **funcs})
    return evaluate


RELATIONS = {'below': lambda v, p: v < p, 'at_most': lambda v, p: v <= p, 'at_least': lambda v, p: v >= p}


def agrees(value, printed, relation='rounds_to'):
    """True if value rounds to the printed decimal or equals the printed fraction p/q; or, for a stated bound ('below',
    'at_most', 'at_least'), if value satisfies it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value): return False
    if relation != 'rounds_to': return RELATIONS[relation](value, float(printed))
    if '/' in printed: return abs(value - Fraction(printed)) <= 1e-12
    places = len(printed.split('.')[1]) if '.' in printed else 0
    return abs(value - float(printed)) <= .5 * 10 ** -places * (1 + 1e-9) + 1e-12


def reported(sup):
    """Yield one message per inventory entry that fails; the inventory must be non-empty with unique keys."""
    entries = json.loads((sup / 'reported_values.json').read_text())['entries']
    keys = [e['key'] for e in entries]
    if not entries or len(set(keys)) != len(keys): yield 'reported_values.json: empty or duplicate keys'
    evaluate = evaluator(sup)
    for e in entries:
        try:
            value = evaluate(e['source'])
        except Exception as exc:  # a missing file, key or record is a failed entry
            yield f"{e['key']}: {type(exc).__name__}: {exc}"; continue
        if not agrees(value, e['printed'], e.get('relation', 'rounds_to')): yield f"{e['key']}: printed {e['printed']} ({e.get('relation', 'rounds_to')}), stored results give {value!r}"


# ---------------------------------------------------------------- comparison
def differences(a, b, where='$'):
    """Yield the paths at which two parsed values differ (numbers within tolerance count as equal)."""
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        if a != b or type(a) is not type(b): yield where
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL): yield where
    elif isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys(): yield where + ' (keys)'
        for k in a.keys() & b.keys(): yield from differences(a[k], b[k], f'{where}.{k}')
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): yield where + ' (length)'
        for i, (x, y) in enumerate(zip(a, b)): yield from differences(x, y, f'{where}[{i}]')
    else:
        yield where


def parse(name, text):
    if name.endswith('.json'): return json.loads(text)
    return [{k: (float(v) if v.replace('.', '', 1).replace('-', '', 1).isdigit() else v) for k, v in r.items()} for r in csv.DictReader(io.StringIO(text))]


def report(title, messages):
    messages = list(messages)
    for m in messages[:20]: print('FAIL       ' + m)
    if len(messages) > 20: print(f'FAIL       ... and {len(messages) - 20} more')
    print(f'{title}: ' + ('passed' if not messages else f'{len(messages)} failure(s)'))
    return len(messages)


def guarded(fn, *args):
    try:
        yield from fn(*args)
    except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:  # e.g. a deleted or unreadable input
        yield f'{fn.__name__}: {type(exc).__name__}: {exc}'


def value_files(sup, computed):
    for name, text in computed.items():
        stored = (sup / name).read_bytes().decode()
        if stored == text:
            print(f'identical  {name}'); continue
        diff = list(differences(parse(name, text), parse(name, stored)))
        if diff: yield f'{name}: {len(diff)} entries differ, e.g. ' + ', '.join(diff[:5])
        else: print(f'equal      {name} (numbers within relative tolerance {REL_TOL:g})')


def recompute_all(sup):
    computed = {}
    for part in (main_study, sixvar_violations, extensions): computed.update(part(sup))
    return computed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--supplement', type=Path, default=ROOT / 'manuscript/supplement', help='folder with the stored results and value files')
    p.add_argument('--data', type=Path, default=ROOT / 'data/raw', help='folder with the raw data (SPHERE files and UCI archive)')
    p.add_argument('--write', type=Path, help='write the recomputed value files to this new directory instead of comparing')
    args = p.parse_args()
    sup = args.supplement.resolve()
    if args.write:
        computed = recompute_all(sup)
        args.write.mkdir(parents=True, exist_ok=False)
        for name, text in computed.items(): (args.write / name).write_bytes(text.encode())
        print(f'wrote {len(computed)} files to {args.write}'); return 0
    failed = report('integrity of the stored files', guarded(integrity, sup, args.data.resolve()))
    try:
        computed = recompute_all(sup)
    except (OSError, KeyError, IndexError, TypeError, ValueError, AssertionError) as exc:
        failed += report('recomputed value files', [f'recomputation: {type(exc).__name__}: {exc}'])
    else:
        failed += report('recomputed value files', guarded(value_files, sup, computed))
    n = len(json.loads((sup / 'reported_values.json').read_text())['entries']) if (sup / 'reported_values.json').is_file() else 0
    failed += report(f'reported numbers ({n} entries)', guarded(reported, sup))
    print('All checks passed: stored files intact, value files recomputed, every reported number matches the stored results.' if not failed else f'{failed} failure(s).')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
