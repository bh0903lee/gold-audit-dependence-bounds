"""SPHERE house PIR record with video-derived location labels (Section 8.4, Supplementary Table S6): descriptive analysis
fixed before it was run (docs/PROTOCOL.md, "SPHERE house recordings").

Data: SPHERE raw view (Tonkin et al. 2023, Sci Data 10:162; doi:10.5523/bris.2h0wyctxrd69j2oqccsi45hy1p, CC BY 4.0),
files data/raw/sphere_house/raw/NNN/{meta.json, pir_raw.csv, location_K_raw.csv} for the 19 sequences.

Definitions : grid of g-second cells [c g, (c+1) g) from the sequence start. For annotator K, the living-room
share of a cell is the overlap of its location_K_raw intervals named R with the cell, divided by g; the labeled share is
the same for intervals of any room. X = 1 if the mean living share over the annotators > 0.5; a cell is unlabeled if the
mean labeled share <= 0.5. Y = 1 if a pir_raw interval named R overlaps the cell. Windows are (c-k, c, c+k) inside one
sequence; thinned windows use centers k, 3k+1, ... (spacing 2k+1 cells, disjoint spans); a window touching an unlabeled
cell is dropped. Record target theta_hat = 1 - 2 * #(X_{c-k} != X_c) / n.

Finite-sample procedure (as in the manuscript Sections 8.1 and 8.4): observation moments from all windows in the
pool; per draw, B audit and B calibration windows without replacement and disjoint; false-positive budget = one-sided
Clopper-Pearson upper limit (0.99) on the calibration windows (budget 0 for the three-state contrast); four-state program
with simultaneous CP moment intervals (alpha 0.04) for M, N, F. 1000 draws, seed 480001 + draw.

Usage (from the package root; data/raw/sphere_house/DATA_SOURCE.md):
  python -I extensions/scripts/sphere_pir.py data/raw/sphere_house/raw OUT_DIR [DRAWS] [WORKERS]
"""
import csv, json, math, sys, hashlib
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from c11_upgrade.population import X, Y  # noqa: E402
from c11_upgrade.addendum_core import population_references, solve_information  # noqa: E402
from c11_upgrade.inference import cp_upper  # noqa: E402

ENCODE = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(X, Y))}
BUDGETS = (50, 100, 200)
PRIMARY = dict(room='living', grid=10, lag=1, thinned=True, y_rule='overlap', shift=0, audit_labels='consensus',
               states=4)
# Sensitivity list fixed before the run (descriptive; no decisions).
SENSITIVITIES = (
    [dict(name=f'grid {g}s lag {k}', grid=g, lag=k) for g, k in ((5, 1), (5, 2), (10, 3), (20, 1), (30, 1))]
    + [dict(name='overlapping windows', thinned=False), dict(name='kitchen', room='kitchen'),
       dict(name='onset-only Y', y_rule='onset')]
    + [dict(name=f'clock shift {d:+d}s', shift=d) for d in (-2, -1, 1, 2)]
    + [dict(name=f'audit labels annotator {a}', audit_labels=a) for a in (0, 1)]
    + [dict(name='three-state (budget 0)', states=3)])


def read_intervals(path, start, end):
    with open(path, newline='') as f:
        return [(r['name'], float(r[start]), float(r[end])) for r in csv.DictReader(f)]


def load(raw):
    seqs = []
    for d in sorted(p for p in Path(raw).iterdir() if p.is_dir()):
        meta = json.loads((d / 'meta.json').read_text())
        anns = sorted(d.glob('location_*_raw.csv'), key=lambda p: int(p.stem.split('_')[1]))
        seqs.append(dict(seq=d.name, user=meta['user_id'], end=float(meta['end']),
                         pir=read_intervals(d / 'pir_raw.csv', 'start', 'end'),
                         loc=[read_intervals(p, 'start', 'end') for p in anns]))
    return seqs


def overlap(intervals, n, g, shift=0.0):
    """Seconds of each cell [c g, (c+1) g) covered by the intervals (merged, clipped to [0, n g))."""
    iv = sorted((max(0.0, s + shift), min(n * g, e + shift)) for s, e in intervals)
    merged = []
    for s, e in iv:
        if e <= s: continue
        if merged and s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([s, e])
    cov = np.zeros(n)
    for s, e in merged:
        for c in range(int(s // g), min(n, int(math.ceil(e / g)))):
            cov[c] += max(0.0, min(e, (c + 1) * g) - max(s, c * g))
    return cov


def cells(seq, room, g, y_rule, shift):
    n = int(seq['end'] // g)
    live = np.array([overlap([(s, e) for nm, s, e in a if nm == room], n, g) / g for a in seq['loc']])
    lab = np.array([overlap([(s, e) for _, s, e in a], n, g) / g for a in seq['loc']])
    x = (live.mean(0) > .5).astype(int); labelled = lab.mean(0) > .5
    x_ann = (live > .5).astype(int)
    pir = [(s, e) for nm, s, e in seq['pir'] if nm == room]
    if y_rule == 'overlap':
        y = (overlap(pir, n, g, shift) > 0).astype(int)
    else:  # onset-only: an activation starts inside the cell
        y = np.zeros(n, int)
        for s, _ in pir:
            c = int((s + shift) // g)
            if 0 <= c < n: y[c] = 1
    bed2 = (overlap([(s, e) for nm, s, e in seq['pir'] if nm == 'bed2'], n, g) > 0).astype(int)
    return dict(x=x, y=y, labelled=labelled, x_ann=x_ann, bed2=bed2)


def build_pool(seqs, cfg):
    k, g = cfg['lag'], cfg['grid']; rows = []
    for seq in seqs:
        cl = cells(seq, cfg['room'], g, cfg['y_rule'], cfg['shift']); n = len(cl['x'])
        centres = range(k, n - k, 2 * k + 1) if cfg['thinned'] else range(k, n - k)
        for c in centres:
            idx = (c - k, c, c + k)
            if not all(cl['labelled'][i] for i in idx): continue
            ann = cfg['audit_labels']
            xa = cl['x'] if ann == 'consensus' else cl['x_ann'][ann]
            rows.append(dict(user=seq['user'], seq=seq['seq'], x=[int(cl['x'][i]) for i in idx],
                             y=[int(cl['y'][i]) for i in idx], x_audit=[int(xa[i]) for i in idx],
                             bed2=int(cl['bed2'][c])))
    return rows


def describe(rows):
    n = len(rows)
    xx = np.array([r['x'] for r in rows]); yy = np.array([r['y'] for r in rows])
    counts = np.zeros((4, 4, 4), int)
    for r in rows: counts[tuple(ENCODE[(x, y)] for x, y in zip(r['x'], r['y']))] += 1
    law = np.array([Fraction(int(v), n) for v in counts.ravel()], dtype=object).reshape(4, 4, 4)
    ref = population_references(law)
    changes = int(np.sum(xx[:, 0] != xx[:, 1]))
    x1, y1 = xx[:, 1], yy[:, 1]
    out = dict(n=n, sequences=len({r['seq'] for r in rows}), theta_record=1 - 2 * changes / n,
               theta_record_lp=ref['theta'], changes_first_pair=changes, f_hat=float(np.mean((x1 == 0) & (y1 == 1))),
               p_y1_given_x0=float(np.mean(y1[x1 == 0])) if np.any(x1 == 0) else None,
               miss_rate=float(np.mean(y1[x1 == 1] == 0)) if np.any(x1 == 1) else None,
               occupied_share=float(np.mean(x1)), recorded_share=float(np.mean(y1)),
               bed2_rate=float(np.mean([r['bed2'] for r in rows])),
               stationarity_residual=ref['stationarity_residual'], exact_reference={})
    for info, iv in ref['intervals'].items():
        e = dict(status=iv['status'])
        if iv['status'] == 'ok':
            e.update(lower=iv['lower'], upper=iv['upper'],
                     record_inside=bool(iv['lower'] - 1e-9 <= out['theta_record'] <= iv['upper'] + 1e-9))
        out['exact_reference'][info] = e
    return out


def wilson_lower(p, n, z=1.959963984540054):
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def finite_sample(args):
    rows, theta, budget, draws, states = args
    yy = np.array([r['y'] for r in rows], int); xa = np.array([r['x_audit'][1] for r in rows], int); n = len(rows)
    acc = {k: dict(contains=0, width=0.0, nonok=0, upper_below_1=0, lower_above_0=0) for k in 'MNF'}
    for r in range(draws):
        order = np.argsort(np.random.default_rng(480001 + r).random(n), kind='stable')
        aud, cal = order[:budget], order[budget:2 * budget]
        eps = cp_upper(int(np.sum((xa[cal] == 0) & (yy[cal, 1] == 1))), budget, .01) if states == 4 else 0.0
        for k, iv in solve_information(yy, xa[aud], yy[aud], eps, .04).items():
            if iv['status'] == 'solver_failure': raise RuntimeError(f'solver failure: draw {r}, {k}, B={budget}')
            lo, hi = (iv['lower'], iv['upper']) if iv['status'] == 'ok' else (-1.0, 1.0)
            a = acc[k]; a['nonok'] += iv['status'] != 'ok'
            a['contains'] += bool(lo - 1e-9 <= theta <= hi + 1e-9); a['width'] += hi - lo
            a['upper_below_1'] += hi < 1 - 1e-9; a['lower_above_0'] += lo > 1e-9
    out = {}
    for k, a in acc.items():
        p = a['contains'] / draws
        out[k] = dict(contains_record_target=p, wilson95_lower=wilson_lower(p, draws),
                      contains_empty_as_miss=(a['contains'] - a['nonok']) / draws, mean_width=a['width'] / draws,
                      nonok=a['nonok'], share_upper_below_1=a['upper_below_1'] / draws,
                      share_lower_above_0=a['lower_above_0'] / draws)
    out['N_vs_M_width_reduction'] = 1 - out['N']['mean_width'] / out['M']['mean_width']
    out['N_vs_F_extra_reduction_by_F'] = 1 - out['F']['mean_width'] / out['N']['mean_width']
    return out


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__ == '__main__':
    raw, outdir = Path(sys.argv[1]), Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
    draws = int(sys.argv[3]) if len(sys.argv) > 3 else 1000; workers = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    seqs = load(raw)
    configs = [dict(name='primary', **PRIMARY)] + [{**PRIMARY, **s} for s in SENSITIVITIES]
    results, jobs = [], []
    for cfg in configs:
        rows = build_pool(seqs, cfg); d = describe(rows)
        results.append(dict(config=cfg, **d, finite_sample={}))
        for b in BUDGETS:
            if len(rows) >= 2 * b: jobs.append((len(results) - 1, b, (rows, d['theta_record'], b, draws, cfg['states'])))
        print(cfg['name'], 'n =', len(rows), flush=True, file=sys.stderr)
    with Pool(workers) as pool:
        for (i, b, _), fs in zip(jobs, pool.map(finite_sample, [j[2] for j in jobs])):
            results[i]['finite_sample'][str(b)] = fs
    primary_rows = build_pool(seqs, configs[0])
    per_participant = {}
    for u in sorted({r['user'] for r in primary_rows}):
        sub = [r for r in primary_rows if r['user'] == u]; xx = np.array([r['x'] for r in sub]); yy = np.array([r['y'] for r in sub])
        per_participant[str(u)] = dict(n=len(sub), theta_record=1 - 2 * float(np.mean(xx[:, 0] != xx[:, 1])),
                                       f_hat=float(np.mean((xx[:, 1] == 0) & (yy[:, 1] == 1))),
                                       miss_rate=float(np.mean(yy[xx[:, 1] == 1, 1] == 0)) if np.any(xx[:, 1] == 1) else None)
    inputs = {str(p.relative_to(raw)): sha256(p) for s in sorted(raw.iterdir()) if s.is_dir()
              for p in sorted(s.iterdir()) if p.name == 'meta.json' or p.name == 'pir_raw.csv' or p.name.startswith('location_')}
    (outdir / 'summary.json').write_text(json.dumps(dict(draws=draws, results=results, per_participant=per_participant,
                                                         input_sha256=inputs), indent=1, default=float))
    lines = []
    for r in results:
        er = r['exact_reference']
        f = lambda k: f"[{er[k]['lower']:.3f},{er[k]['upper']:.3f}]{'in' if er[k]['record_inside'] else 'OUT'}" if er[k]['status'] == 'ok' else er[k]['status']
        lines.append(f"{r['config']['name']:30s} n={r['n']:5d} theta={r['theta_record']:.4f} chg={r['changes_first_pair']} "
                     f"f={r['f_hat']:.4f} P(Y1|X0)={r['p_y1_given_x0']:.3f} miss={r['miss_rate']:.3f} occ={r['occupied_share']:.3f} "
                     f"bed2={r['bed2_rate']:.4f} res={r['stationarity_residual']:.4f} exact M={f('M')} N={f('N')} F={f('F')}")
        for b, fs in r['finite_sample'].items():
            lines.append(f"   B={b:>3s}: " + '  '.join(
                f"{k}: cont {fs[k]['contains_record_target']:.3f} (W {fs[k]['wilson95_lower']:.3f}; empty=miss {fs[k]['contains_empty_as_miss']:.3f}) w {fs[k]['mean_width']:.3f} "
                f"up<1 {fs[k]['share_upper_below_1']:.2f} lo>0 {fs[k]['share_lower_above_0']:.2f} nonok {fs[k]['nonok']}" for k in 'MNF')
                + f"  N/M red {fs['N_vs_M_width_reduction']:.3f}")
    lines.append('per participant: ' + json.dumps(per_participant))
    (outdir / 'summary.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
