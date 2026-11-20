"""Recorded-data study (article Section 9, Supplementary Section S12).

Six data sets with a cheap binary record Y and a reference state X on every occasion:
  prism, prism2, border       malaria cohorts; Y = parasites by microscopy, X = parasites by a molecular test, routine visits
                              four weeks apart; unit = household
  channel_islands, caltech    camera traps; Y = a published detector reports an animal with confidence at least 0.9,
                              X = a human label names an animal, occasions = consecutive trigger sequences; unit = camera
  offices                     two offices; Y = motion counter positive in a five-minute interval, X = a person in the labelled
                              image at the start or end of the interval; unit = one room on one day
Windows are three consecutive occasions of one series. Every procedure receives the record of all windows and the reference
state of `labels` windows; the remaining reference states are used only to compute the target, the lag-one correlation of
the reference state pooled over the two adjacent pairs.

Procedures on the same data, without a model of the latent dynamics or of the detector (M, N, F: marginal, neighbor and
full audit information):
  finite-sample    four-state word program with simultaneous Clopper-Pearson moment intervals and a learned false-positive
                   budget (half of the labels audit, half calibrate), as in Sections 7.2 and 8.6
  percentile       unit-bootstrap percentile interval of the plug-in identified interval
  Imbens-Manski    Imbens-Manski interval of the plug-in identified interval with unit-bootstrap standard errors
  plug-in          the plug-in identified interval itself (no allowance for sampling error)
Procedures with a model: four hidden Markov models (maximum likelihood, unit-bootstrap percentile intervals).
Other: the record alone; an audit of two adjacent occasions (both with unit-bootstrap percentile intervals).

Intervals are evaluated after resampling units with replacement; an empty confidence set counts as a failure.
The correlation range of a confidence set is optimized jointly over the target and the prevalence. Population intervals
are computed at the least-squares projection of the window law onto the stationary laws whose false-positive mass does
not exceed that of the data (total-variation projection as a sensitivity analysis). Two stationary laws with the same law of the record and of the audited middle state as the data
and the smallest and the largest target give the worst case for every procedure that uses only those data.

Data: see data/raw/recorded_data/DATA_SOURCE.md (not redistributed). Usage, from the package root:
  python cohorts/cohort_study.py prism2 --output results/generated/cohorts --workers 38
"""
import argparse, collections, csv, datetime as dt, io, itertools, json, os, sys, time, zipfile
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'extensions/src'))
from c11_upgrade.addendum_core import information_limits, make_problem      # noqa: E402
from c11_upgrade.population import X, Y, features, lp_interval              # noqa: E402
from c11_upgrade.inference import cp_upper                                  # noqa: E402

DATA = Path(os.environ.get('RECORDED_DATA', ROOT / 'data/raw/recorded_data'))
csv.field_size_limit(10 ** 8)
ALPHA = .05; INFOS = ('M', 'N', 'F'); BOOT = 200

# ---------------------------------------------------------------- word features (three-point windows, four states)
WORDS, ROWS, FLOW, TARGET = features(3, 4); TARGET = TARGET.astype(float); NW = len(TARGET)
CODE = np.zeros((2, 2), int)
for i, (xx, yy) in enumerate(zip(X, Y)): CODE[xx, yy] = i
WX, WY = X[WORDS], Y[WORDS]
M_ROW, FP_ROW = ROWS['m'].astype(float), ROWS['fp'].astype(float)
OBS = [ROWS[f'o{k}'].astype(float) for k in range(8)]
AUDIT = {'M': [M_ROW], 'N': [M_ROW, ROWS['a'].astype(float), ROWS['b'].astype(float)], 'F': [ROWS[f'u{k}'].astype(float) for k in range(8, 16)]}
GMAT = {k: np.array(OBS + v) for k, v in AUDIT.items()}
EQ = np.r_[FLOW.astype(float), [np.ones(NW)]]; EB = np.r_[np.zeros(len(FLOW)), 1.]
POW = np.array([4, 2, 1])


def window_law(x, y):
    c = CODE[x, y]; return np.bincount(c[:, 0] * 16 + c[:, 1] * 4 + c[:, 2], minlength=NW) / len(x)


def corr_of(theta, m):
    return (theta - (2 * m - 1) ** 2) / (4 * m * (1 - m))


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 and a.std() > 0 and b.std() > 0 else None


def pooled(v):
    """Lag-one correlation pooled over the two adjacent pairs of a window."""
    return phi(np.r_[v[:, 0], v[:, 1]], np.r_[v[:, 1], v[:, 2]])


# ---------------------------------------------------------------- data
def _date(s): return dt.date.fromisoformat(s)


def prism2_visits():
    d = DATA / 'prism2'
    rd = lambda n: list(csv.DictReader(open(d / f'PRISM2_cohort_{n}.txt', encoding='utf-8', newline=''), delimiter='\t'))
    S, R = rd('Samples'), rd('Participant_repeated_measures')
    def col(row, prefix):
        hits = [c for c in row if c.startswith(prefix)]
        if len(hits) != 1: raise KeyError((prefix, hits))
        return hits[0]
    cm, cq, cd = col(S[0], 'Plasmodium asexual stages, by microscopy ['), col(S[0], 'Plasmodium, by qPCR ['), col(S[0], 'Plasmodium, by qPCR result')
    sample = {r['Participant_repeated_measure_Id']: r for r in S}
    if len(sample) != len(S): raise ValueError('several samples for one observation')
    cdate, ctype, cage = col(R[0], 'Observation date'), col(R[0], 'Observation type'), col(R[0], 'Age (years)')
    out = collections.defaultdict(list)
    for r in R:
        q = sample.get(r['Participant_repeated_measure_Id'])
        if q is None or r[ctype] != 'Routine' or q[cm] not in ('Yes', 'No') or q[cq] not in ('Positive', 'Negative') or q[cd] == '': continue
        out[r['Participant_Id']].append(dict(pid=r['Participant_Id'], hh=r['Household_Id'], district='', date=_date(r[cdate]), age=float(r[cage]),
                                             y=int(q[cm] == 'Yes'), q=int(q[cq] == 'Positive'), density=float(q[cd])))
    return dict(out), {}


def prism_visits():
    """First PRISM cohort: loop-mediated isothermal amplification (LAMP) was run on the microscopy-negative samples, so the reference is microscopy or LAMP."""
    d = DATA / 'prism'
    rd = lambda n: list(csv.DictReader(open(d / f'PRISM_cohort_{n}.txt', encoding='utf-8', newline=''), delimiter='\t'))
    S, R, H = rd('Samples'), rd('Participant_repeated_measures'), rd('Households')
    def col(row, prefix):
        hits = [c for c in row if c.startswith(prefix)]
        if len(hits) != 1: raise KeyError((prefix, hits))
        return hits[0]
    cm, cl = col(S[0], 'Plasmodium asexual stages, by microscopy ['), col(S[0], 'Plasmodium, by LAMP'); site = {h['Household_Id']: h[col(H[0], 'Sub-county in Uganda')] for h in H}
    sample = {r['Participant_repeated_measure_Id']: r for r in S}
    if len(sample) != len(S): raise ValueError('several samples for one observation')
    cdate, ctype, cage = col(R[0], 'Observation date'), col(R[0], 'Observation type'), col(R[0], 'Age (years)')
    out = collections.defaultdict(list)
    for r in R:
        q = sample.get(r['Participant_repeated_measure_Id'])
        if q is None or r[ctype] != 'Scheduled visit' or q[cm] not in ('Positive', 'Negative') or (q[cm] == 'Negative' and q[cl] not in ('Positive', 'Negative')): continue
        y = int(q[cm] == 'Positive')
        out[r['Participant_Id']].append(dict(pid=r['Participant_Id'], hh=r['Household_Id'], district=site[r['Household_Id']], date=_date(r[cdate]), age=float(r[cage]), y=y, q=int(y or q[cl] == 'Positive'), density=1.))
    return dict(out), {}


def border_visits():
    R = list(csv.DictReader(open(DATA / 'border/PRISM2_Border_Cohort_All_Visits_database.csv', encoding='utf-8', newline='')))
    out, treated = collections.defaultdict(list), collections.defaultdict(list)
    for r in R:
        d = _date(r['date_obf'])
        if r['malariacat'] == '3': treated[r['cohortid']].append(d)                       # visit with a diagnosis of malaria, which is treated
        if r['visittype'] == '1' and r['BSdich'] in ('0', '1') and r['qPCRdich'] in ('0', '1'):
            out[r['cohortid']].append(dict(pid=r['cohortid'], hh=r['hhid'], district=r['District'], date=d, age=float(r['ageyrs']) if r['ageyrs'] not in ('', 'NA') else float('nan'),
                                           y=int(r['BSdich']), q=int(r['qPCRdich']), density=float(r['qpcr']) if r['qpcr'] not in ('', 'NA') else 0.))
    return dict(out), dict(treated)


CAMERA_FILES = {'caltech': ('caltech_camera_traps.json.zip', 'caltech-camera-traps_mdv5a.0.0_results.json.zip'),
                'channel_islands': ('channel-islands-camera-traps.json.zip', 'channel-islands-camera-traps_mdv5a.0.0_results.json.zip')}
NOT_ANIMAL = ('empty', 'human', 'person', 'car', 'vehicle', 'other', '')
THRESHOLD = .9


def camera_sequences(name):
    """Camera traps: unit = camera location, time step = one trigger sequence in recorded order. X = a human label names an
    animal in the sequence; Y = the published detector output reports an animal with confidence at least THRESHOLD."""
    def member(z):
        with zipfile.ZipFile(DATA / 'camera_traps' / z) as f: return json.loads(f.read(f.namelist()[0]))
    M, R = member(CAMERA_FILES[name][0]), member(CAMERA_FILES[name][1]); cats = {c['id']: c['name'] for c in M['categories']}
    animal = [k for k, v in R['detection_categories'].items() if v == 'animal'][0]
    conf = {im['file']: max([d['conf'] for d in (im.get('detections') or []) if d['category'] == animal], default=0.) for im in R['images']}
    label = collections.defaultdict(set)
    for a in M['annotations']: label[a['image_id']].add(cats[a['category_id']])
    seq = {}
    for im in sorted(M['images'], key=lambda im: (str(im['location']), str(im.get('date_captured', '')), im['file_name'])):
        if im['file_name'] not in conf: continue
        v = seq.setdefault((str(im['location']), im['seq_id']), [0, 0.])
        v[0] = max(v[0], int(any(l not in NOT_ANIMAL for l in label.get(im['id'], ())))); v[1] = max(v[1], conf[im['file_name']])
    out = collections.defaultdict(list)
    for (loc, _), (x, c) in seq.items():
        out[loc].append(dict(pid=loc, hh=loc, district=None, date=dt.date(2000, 1, 1) + dt.timedelta(days=len(out[loc])), age=0., y=int(c >= THRESHOLD), q=x, density=1., conf=c))
    return dict(out), {}


def office_intervals():
    """Two offices: unit = one room on one day, time step = five minutes. Y = the motion counter is positive in the interval;
    X = a person is in the manually labelled image at the start or at the end of the interval."""
    out = collections.defaultdict(list)
    with zipfile.ZipFile(DATA / 'graz_offices' / 'occupancy-detection-dataset.zip') as z:
        for room in 'AB':
            rd = lambda n: list(csv.DictReader(io.StringIO(z.read(f'data/intermi/Room_{room}/{n}.csv').decode('utf-8'))))
            people = {r['Date']: r['Image-based-occupancy'] for r in rd('df_occ')}; previous = None
            for r in rd('df_features'):
                t = dt.datetime.fromisoformat(r['Date']); now = people.get(r['Date']); now = None if now in (None, '', 'nan') else int(float(now) > 0)
                if now is not None and previous is not None and previous[0] == t - dt.timedelta(minutes=5) and r['motion'] not in ('', 'nan'):
                    unit = f'{room} {t.date()}'; c = float(r['motion'])
                    out[unit].append(dict(pid=unit, hh=unit, district=room, date=dt.date(2000, 1, 1) + dt.timedelta(days=(t.hour * 60 + t.minute) // 5), age=float(t.hour), y=int(c > 0),
                                          q=int(now or previous[1]), end=now, density=1., conf=c))
                previous = None if now is None else (t, now)
    return dict(out), {}


INPUTS = {'prism': [f'prism/PRISM_cohort_{n}.txt' for n in ('Samples', 'Participant_repeated_measures', 'Households')],
          'prism2': ['prism2/PRISM2_cohort_Samples.txt', 'prism2/PRISM2_cohort_Participant_repeated_measures.txt'], 'border': ['border/PRISM2_Border_Cohort_All_Visits_database.csv'],
          'channel_islands': [f'camera_traps/{f}' for f in CAMERA_FILES['channel_islands']], 'caltech': [f'camera_traps/{f}' for f in CAMERA_FILES['caltech']],
          'offices': ['graz_offices/occupancy-detection-dataset.zip']}
LOADERS = {'offices': office_intervals, 'caltech': lambda: camera_sequences('caltech'), 'channel_islands': lambda: camera_sequences('channel_islands'), 'prism': prism_visits, 'prism2': prism2_visits, 'border': border_visits}
BASE = dict(sampling='all', gap=(21, 35), lag=1, start=dt.date(2000, 1, 1), end=dt.date(9999, 12, 31), age=(0, 200), district=None, reference='qpcr', untreated=False, threshold=None)
def _c(**k): return {**BASE, **k}
def _year(y): return dict(start=dt.date(y, 1, 1), end=dt.date(y, 12, 31))
COMMON = {'all windows': _c(), 'one window per participant': _c(sampling='participant'), 'one window per household': _c(sampling='household'), 'visits 24-32 days apart': _c(gap=(24, 32)),
          'age under 5 years': _c(age=(0, 5)), 'age 5-15 years': _c(age=(5, 16)), 'age 16 years or older': _c(age=(16, 200)),
          'reference: qPCR or microscopy': _c(reference='union'), 'reference: qPCR density at least 0.1/uL': _c(reference='density'), 'visits eight weeks apart': _c(lag=2)}
SITES = {f'{s} site': dict(district=s) for s in ('Kihihi', 'Nagongera', 'Walukuba')}
STEP = dict(gap=(1, 1))
CAMERA = {'all windows': _c(**STEP), 'one window per participant': _c(**STEP, sampling='participant'), 'every second sequence': _c(**STEP, lag=2),
          **{f'detector threshold {h:.2f}': _c(**STEP, threshold=h) for h in (.8, .85, .95)}}
OFFICE = {'all windows': _c(**STEP), 'one window per participant': _c(**STEP, sampling='participant'), 'room A': _c(**STEP, district='A'), 'room B': _c(**STEP, district='B'),
          'between 04:00 and 17:00 UTC': _c(**STEP, age=(4, 17)), 'occupancy at the end of the interval': _c(**STEP, reference='end'), 'motion count at least 2': _c(**STEP, threshold=2),
          'every second interval': _c(**STEP, lag=2)}
CONFIGS = {'offices': OFFICE, 'caltech': CAMERA, 'channel_islands': CAMERA, 'prism': {**{k: v for k, v in COMMON.items() if not k.startswith(('reference', 'visits eight'))}, **{k: _c(**v) for k, v in SITES.items()}},
           'prism2': {**COMMON, 'windows in 2018': _c(**_year(2018)), 'windows in 2019': _c(**_year(2019))},
           'border': {**COMMON, 'Busia district': _c(district='1'), 'Tororo district': _c(district='2'), 'windows in 2021': _c(**_year(2021)), 'windows in 2022': _c(**_year(2022)),
                      'no malaria episode within the window': _c(untreated=True)}}
REFERENCE = {'end': lambda v: v['end'], 'qpcr': lambda v: v['q'], 'union': lambda v: int(v['q'] or v['y']), 'density': lambda v: int(v['q'] and v['density'] >= .1)}


def describe(visits):
    V = [v for l in visits.values() for v in l]; gaps = [(b['date'] - a['date']).days for l in visits.values() for a, b in zip(sorted(l, key=lambda v: v['date']), sorted(l, key=lambda v: v['date'])[1:])]
    y, q = np.array([v['y'] for v in V]), np.array([v['q'] for v in V])
    return dict(routine_visits=len(V), participants=len(visits), households=len({v['hh'] for v in V}), record_positive=int(y.sum()), record_positive_reference_negative=int(((y == 1) & (q == 0)).sum()),
                reference_positive=int(q.sum()), reference_positive_record_negative=int(((q == 1) & (y == 0)).sum()), median_days_between_visits=float(np.median(gaps)), share_of_gaps_21_to_35_days=float(np.mean([21 <= g <= 35 for g in gaps])))


def build(data, cfg, seed=20261118):
    """Window pool of one configuration: arrays x, y (n x 3) and household labels."""
    visits, treated = data; rng = np.random.default_rng(seed); k = cfg['lag']; ref = REFERENCE[cfg['reference']]; chosen = []
    rec = (lambda v: v['y']) if cfg['threshold'] is None else (lambda v: int(v['conf'] >= cfg['threshold']))
    for pid in sorted(visits):
        l = sorted(visits[pid], key=lambda v: v['date']); cand = []
        if cfg['district'] and l[0]['district'] != cfg['district']: continue
        for i in range(len(l) - 2 * k):
            a, b, c = l[i], l[i + k], l[i + 2 * k]
            if not (cfg['gap'][0] * k <= (b['date'] - a['date']).days <= cfg['gap'][1] * k and cfg['gap'][0] * k <= (c['date'] - b['date']).days <= cfg['gap'][1] * k): continue
            if not (cfg['start'] <= a['date'] and c['date'] <= cfg['end'] and cfg['age'][0] <= b['age'] < cfg['age'][1]): continue
            if cfg['untreated'] and any(a['date'] <= d <= c['date'] for d in treated.get(pid, ())): continue
            cand.append((i, a, b, c))
        if not cand: continue
        if cfg['sampling'] == 'all':
            last = -10 ** 9
            for t in cand:
                if t[0] > last: chosen.append(t); last = t[0] + 2 * k
        else: chosen.append(cand[int(rng.integers(len(cand)))])
    if cfg['sampling'] == 'household':
        by = collections.defaultdict(list)
        for t in chosen: by[t[1]['hh']].append(t)
        chosen = [by[h][int(rng.integers(len(by[h])))] for h in sorted(by)]
    hh = np.array([t[1]['hh'] for t in chosen]); _, lab = np.unique(hh, return_inverse=True)
    return dict(x=np.array([[ref(v) for v in t[1:]] for t in chosen], int), y=np.array([[rec(v) for v in t[1:]] for t in chosen], int), hh=lab, participants=len({t[1]['pid'] for t in chosen}))


def resample(pool, rng):
    """Households drawn with replacement; each draw is a new cluster."""
    g = pool.setdefault('groups', [np.flatnonzero(pool['hh'] == h) for h in range(pool['hh'].max() + 1)])
    pick = rng.integers(len(g), size=len(g)); take = np.concatenate([g[i] for i in pick])
    return pool['x'][take], pool['y'][take], np.repeat(np.arange(len(pick)), [len(g[i]) for i in pick])


# ---------------------------------------------------------------- population intervals at one stationary law
_AFFINE = EQ.T @ np.linalg.pinv(EQ @ EQ.T)


FP_ALL = np.array([((WX[:, k] == 0) & (WY[:, k] == 1)).astype(float) for k in range(3)])     # false-positive indicator at each window position


def project_ls(p, iterations=400000, tol=1e-13):
    """Least-squares projection onto the stationary laws whose false-positive mass does not exceed that of p (mean over the
    three window positions): Dykstra's alternating projections onto the affine constraints, the nonnegative orthant and the
    half-space of the false-positive mass. A one-sided law is projected onto the one-sided stationary laws."""
    q = np.array(p, float); cap = float((FP_ALL @ q).mean()); a = FP_ROW; aa = float(a @ a); c1 = np.zeros(NW); c2 = np.zeros(NW)
    for _ in range(iterations):
        z = q - _AFFINE @ (EQ @ q - EB)
        y1 = z + c1; u = np.clip(y1, 0, None); c1 = y1 - u
        y2 = u + c2; new = y2 - max(0., float(a @ y2) - cap) / aa * a; c2 = y2 - new
        if np.abs(new - q).max() < tol and np.abs(EQ @ new - EB).max() < 1e-11 and new.min() > -1e-12 and a @ new <= cap + 1e-12: q = np.clip(new, 0, None); break
        q = new
    else: raise RuntimeError('projection did not converge')
    return q / q.sum()


def project_tv(p):
    I = np.eye(NW); r = linprog(np.r_[np.zeros(NW), np.ones(NW)], A_ub=np.r_[np.c_[I, -I], np.c_[-I, -I]], b_ub=np.r_[p, -p], A_eq=np.c_[EQ, np.zeros((len(EQ), NW))], b_eq=EB,
                                bounds=[(0, None)] * (2 * NW), method='highs')
    if r.status != 0: raise RuntimeError(r.message)
    return r.x[:NW]


def _extremes(q, info, tol=1e-9):
    g = GMAT[info] @ q; f = float(FP_ROW @ q); A = np.r_[GMAT[info], -GMAT[info], [FP_ROW]]; b = np.r_[g + tol, -g + tol, f + tol]; out = []
    for s in (1, -1):
        r = linprog(s * TARGET, A_ub=A, b_ub=b, A_eq=EQ, b_eq=EB, bounds=(0, None), method='highs')
        if r.status != 0: raise RuntimeError(r.message)
        out.append((s * float(r.fun), np.clip(r.x, 0, None) / np.clip(r.x, 0, None).sum()))
    return out


def sharp(q):
    """Sharp four-state intervals of the target at the moments of one stationary law (false-positive mass at its value in that law)."""
    m = float(M_ROW @ q); out = {}
    for info in INFOS:
        (lo, _), (hi, _) = _extremes(q, info)
        out[info] = dict(theta_lower=lo, theta_upper=hi, theta_width=hi - lo, lower=corr_of(lo, m), upper=corr_of(hi, m), width=corr_of(hi, m) - corr_of(lo, m))
    return out


def equivalent_laws(q):
    """Two stationary laws with the full audit information of q and the smallest and the largest target."""
    (_, a), (_, b) = _extremes(q, 'F', tol=1e-10); law_corr = lambda p: corr_of(float(TARGET @ p), float(M_ROW @ p))
    return a, b, law_corr(a), law_corr(b), float(np.abs(GMAT['F'] @ a - GMAT['F'] @ b).max())


# ---------------------------------------------------------------- intervals without a model
def finite_sample(y, ax, ay, epsilon, alpha, info, cells=48):
    """Outer correlation range of the confidence set of the word program; None if the set is empty.

    The correlation is 1 - (1 - theta) / u with u = 4 m (1 - m), increasing in theta and in u. The prevalence range of the
    confidence set is cut into cells; in each cell the largest (smallest) target is paired with the largest (smallest) u of the
    cell, which bounds the correlation over the cell from above (below). The union over cells contains the exact range."""
    p = make_problem(information_limits(y, ax, ay, info, alpha), epsilon, info)
    if lp_interval(p)['status'] != 'ok': return None
    Ae, be, Au, bu, c = (np.asarray(v, float) for v in (p.ae, p.be, p.au, p.bu, p.c))
    lim = [linprog(s * M_ROW, A_ub=Au, b_ub=bu, A_eq=Ae, b_eq=be, bounds=(0, None), method='highs') for s in (1, -1)]
    if any(r.status != 0 for r in lim): return None
    lo, hi = float(lim[0].fun), float(-lim[1].fun); u = lambda m: 4 * m * (1 - m); edges = np.linspace(lo, hi, cells + 1) if hi - lo > 1e-9 else np.array([lo, hi]); low, high = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        A = np.r_[Au, [M_ROW, -M_ROW]]; rhs = np.r_[bu, b + 1e-12, -a + 1e-12]
        r = [linprog(s * c, A_ub=A, b_ub=rhs, A_eq=Ae, b_eq=be, bounds=(0, None), method='highs') for s in (1, -1)]
        if any(q.status != 0 for q in r): continue
        tmin, tmax = float(r[0].fun), float(-r[1].fun); umax = 1. if a <= .5 <= b else max(u(a), u(b)); umin = min(u(a), u(b))
        high.append(1 - (1 - tmax) / umax if umax > 0 else 1.); low.append(1 - (1 - tmin) / umin if umin > 1e-12 else -1.)
    return (float(np.clip(min(low), -1, 1)), float(np.clip(max(high), -1, 1))) if low else None


PLUG = {k: (np.r_[np.c_[G, -np.ones(len(G))], np.c_[-G, -np.ones(len(G))], [np.r_[FP_ROW, -1.]]], np.c_[EQ, np.zeros(len(EQ))]) for k, G in GMAT.items()}


def sample_moments(y, ax, ay, info):
    obs = np.bincount(y @ POW, minlength=8) / len(y); pat = ay @ POW
    aud = {'M': [ax.mean()], 'N': [ax.mean(), np.mean(ax * ay[:, 0]), np.mean(ax * ay[:, 2])], 'F': [np.mean(ax * (pat == k)) for k in range(8)]}[info]
    return np.r_[obs, aud]


def plug_in(y, ax, ay, info):
    """Identified interval at the sample moments, relaxed uniformly by the smallest amount that a stationary law admits."""
    g = sample_moments(y, ax, ay, info); f = float(np.mean((ax == 0) & (ay[:, 1] == 1))); m = float(ax.mean()); Aub, Aeq = PLUG[info]; b = np.r_[g, -g, f]
    if not 0 < m < 1: return None
    r = linprog(np.r_[np.zeros(NW), 1.], A_ub=Aub, b_ub=b, A_eq=Aeq, b_eq=EB, bounds=[(0, None)] * (NW + 1), method='highs')
    if r.status != 0: return None
    t = r.x[-1] * (1 + 1e-6) + 1e-10; e = []
    for s in (1, -1):
        q = linprog(np.r_[s * TARGET, 0.], A_ub=Aub, b_ub=b, A_eq=Aeq, b_eq=EB, bounds=[(0, None)] * NW + [(t, t)], method='highs')
        if q.status != 0: return None
        e.append(float(np.clip(corr_of(s * float(q.fun), m), -1, 1)))
    return e[0], e[1]


def imbens_manski(lower, upper, se_lower, se_upper, alpha=ALPHA):
    d = max(upper - lower, 0.); s = max(se_lower, se_upper, 1e-9); lo, hi = 1., 3.
    for _ in range(50):
        c = (lo + hi) / 2
        if norm.cdf(c + d / s) - norm.cdf(-c) < 1 - alpha: lo = c
        else: hi = c
    return max(lower - hi * se_lower, -1.), min(upper + hi * se_upper, 1.)


def large_sample(x, y, hh, audited, rng):
    """Plug-in, percentile and Imbens-Manski intervals; the bootstrap resamples households and keeps each window's audit status."""
    out = {}; flag = np.zeros(len(x), bool); flag[audited] = True; groups = [np.flatnonzero(hh == h) for h in range(hh.max() + 1)]
    draws = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]) for _ in range(BOOT)]
    for k in INFOS:
        p = plug_in(y, x[audited, 1], y[audited], k); b = []
        if p is None:
            out.update({f'{n} {k}': None for n in ('plug-in', 'percentile', 'Imbens-Manski')}); continue
        for t in draws:
            a = t[flag[t]]
            if len(a) >= 20:
                q = plug_in(y[t], x[a, 1], y[a], k)
                if q is not None: b.append(q)
        b = np.array(b); out[f'plug-in {k}'] = p; out[f'percentile {k}'] = (float(np.percentile(b[:, 0], 100 * ALPHA / 2)), float(np.percentile(b[:, 1], 100 * (1 - ALPHA / 2))))
        out[f'Imbens-Manski {k}'] = imbens_manski(p[0], p[1], float(b[:, 0].std(ddof=1)), float(b[:, 1].std(ddof=1)))
    return out


# ---------------------------------------------------------------- hidden Markov models
LATENT = np.array(list(itertools.product((0, 1), repeat=4)))          # X_{-1}, X_0, X_1, X_2
PATTERNS = np.array(list(itertools.product((0, 1), repeat=3)))
MIDDLE = np.array([LATENT[:, 2] == 0, LATENT[:, 2] == 1], float)


def cell_probabilities(m, rho, d0, d1, fp):
    """P(X_1 = x, recorded pattern) as a 2 x 8 array for a stationary two-state chain with prevalence m and lag-one correlation rho.
    d0, d1: detection probability when the previous latent state is 0 or 1; fp: P(Y = 1 | X = 0)."""
    a, b = m + rho * (1 - m), m * (1 - rho); tr = np.array([[1 - b, b], [1 - a, a]]); det = np.array([d0, d1])
    w = np.where(LATENT[:, 0] == 1, m, 1 - m) * tr[LATENT[:, 0], LATENT[:, 1]] * tr[LATENT[:, 1], LATENT[:, 2]] * tr[LATENT[:, 2], LATENT[:, 3]]
    p1 = np.stack([np.where(LATENT[:, i + 1] == 1, det[LATENT[:, i]], fp) for i in range(3)], 1)                    # 16 x 3
    e = np.prod(np.where(PATTERNS[:, None, :] == 1, p1[None], 1 - p1[None]), axis=2)                                # 8 x 16
    return (MIDDLE * w) @ e.T


def log_likelihood(unaudited, audited, cells):
    p = np.clip(cells, 1e-300, None); return float(unaudited @ np.log(p.sum(0)) + (audited * np.log(p)).sum())


def chain(p01, p11): return p01 / (1 - p11 + p01), p11 - p01                 # prevalence and lag-one correlation of a two-state chain


EDGE = (1e-4, 1 - 1e-4)
HMM = {'HMM, one-sided error': (lambda t: (t[0], t[0], 1e-6), [EDGE]),
       'HMM, false positives': (lambda t: (t[0], t[0], t[1]), [EDGE, (1e-6, .2)]),
       'HMM, detection depends on previous state': (lambda t: (t[0], t[1], 1e-6), [EDGE, EDGE])}


def hmm_fit(name, unaudited, audited, starts):
    """Maximum-likelihood fit; parameters are the two transition probabilities P(1 | 0), P(1 | 1) and the detector."""
    spec, box = HMM[name]; best = None
    for x0 in starts:
        r = minimize(lambda t: -log_likelihood(unaudited, audited, cell_probabilities(*chain(t[0], t[1]), *spec(t[2:]))), x0, method='L-BFGS-B', bounds=[EDGE, EDGE] + box, options=dict(maxiter=300))
        if best is None or r.fun < best.fun: best = r
    return best.x


CLASS_BOX = [(.02, .98), EDGE, EDGE, EDGE, EDGE, EDGE, EDGE, (1e-6, .2)]   # weight, P(1 | 0) and P(1 | 1) of each class, d0, d1, fp


def class_cells(t): return t[0] * cell_probabilities(*chain(t[1], t[2]), t[5], t[6], t[7]) + (1 - t[0]) * cell_probabilities(*chain(t[3], t[4]), t[5], t[6], t[7])


def class_corr(t):
    w = (t[0], 1 - t[0]); c = [chain(t[1], t[2]), chain(t[3], t[4])]; m = sum(a * b for a, (b, _) in zip(w, c)); p11 = sum(a * (b * b + r * b * (1 - b)) for a, (b, r) in zip(w, c))
    return (p11 - m * m) / (m * (1 - m))


def class_fit(unaudited, audited, starts):
    best = None
    for x0 in starts:
        r = minimize(lambda t: -log_likelihood(unaudited, audited, class_cells(t)), x0, method='L-BFGS-B', bounds=CLASS_BOX, options=dict(maxiter=300))
        if best is None or r.fun < best.fun: best = r
    return best.x


def model_intervals(y, x_audited, audited, hh, rng, boots=(200, 100)):
    """Hidden Markov models fitted by maximum likelihood to the recorded patterns of the unaudited windows and the audited
    state with the recorded pattern of the audited windows. Percentile intervals; the bootstrap resamples units and refits."""
    flag = np.zeros(len(y), bool); flag[audited] = True; cells = y @ POW; cells[audited] = 8 + 8 * x_audited + cells[audited]
    split = lambda c: (c[:8], c[8:].reshape(2, 8)); total = np.bincount(cells, minlength=24).astype(float); draws = cluster_counts(cells, hh, 24, rng, boots[0]); out = {}
    est = {}
    for name, (spec, box) in HMM.items():
        extra = [.01 if hi < .5 else .5 for _, hi in box]; t = hmm_fit(name, *split(total), [np.r_[a, b, extra] for a, b in ((.2, .6), (.05, .9), (.4, .5))]); est[name] = float(chain(t[0], t[1])[1])
        v = [chain(*hmm_fit(name, *split(d), [t])[:2])[1] for d in draws]; out[name] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    starts = [np.array([rng.uniform(.1, .9)] + [rng.uniform(.05, .95) for _ in range(6)] + [.01]) for _ in range(4)]; t = class_fit(*split(total), starts)
    v = [class_corr(class_fit(*split(d), [t])) for d in draws[:boots[1]]]; out['HMM, two latent classes'] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))); est['HMM, two latent classes'] = float(class_corr(t))
    return out, est


# ---------------------------------------------------------------- record alone, adjacent-pair audit
def pattern_corr(counts):
    """Pooled lag-one correlation of a binary series from counts of its eight window patterns (last axis)."""
    P = np.array(PATTERNS); n11 = counts @ (P[:, 0] * P[:, 1] + P[:, 1] * P[:, 2]); a = counts @ (P[:, 0] + P[:, 1]); b = counts @ (P[:, 1] + P[:, 2]); N = 2 * counts.sum(-1)
    d = a * (N - a) * b * (N - b); return np.where(d > 0, (N * n11 - a * b) / np.sqrt(np.where(d > 0, d, 1)), np.nan)


def cluster_counts(cells, hh, size, rng, reps):
    """Cell counts in bootstrap samples that draw whole units (households, cameras, room-days) with replacement."""
    _, g = np.unique(hh, return_inverse=True)
    if g.max() + 1 == len(g): c = np.bincount(cells, minlength=size); return rng.multinomial(len(g), c / len(g), size=reps).astype(float)      # every unit is one window
    C = np.zeros((g.max() + 1, size)); np.add.at(C, (g, cells), 1)
    return rng.multinomial(len(C), np.full(len(C), 1 / len(C)), size=reps) @ C


def pair_interval(a, b, hh, rng, reps=2000):
    """Percentile interval of the correlation of two binary variables; the bootstrap resamples units."""
    n00, n01, n10, n11 = cluster_counts(2 * np.asarray(a, int) + np.asarray(b, int), hh, 4, rng, reps).T
    d = (n10 + n11) * (n00 + n01) * (n01 + n11) * (n00 + n10); ok = d > 0
    if ok.mean() < .9: return -1., 1.
    v = (n11[ok] * n00[ok] - n10[ok] * n01[ok]) / np.sqrt(d[ok]); return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ---------------------------------------------------------------- one evaluation
def evaluate(x, y, hh, labels, rng, models=True):
    """Intervals of every procedure for the lag-one correlation; None marks an empty confidence set."""
    n = len(x); order = rng.permutation(n); out = {}
    aud, cal = order[:labels // 2], order[labels // 2:labels]; eps = cp_upper(int(np.sum((x[cal, 1] == 0) & (y[cal, 1] == 1))), len(cal), .01)
    for k in INFOS: out[f'finite-sample {k}'] = finite_sample(y, x[aud, 1], y[aud], eps, .04, k)
    lab = order[:labels]; out.update(large_sample(x, y, hh, lab, rng))
    if not models: return out
    bs = pattern_corr(cluster_counts(y @ POW, hh, 8, rng, 1000)); bs = bs[~np.isnan(bs)]
    out['record alone'] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    pair = order[:labels // 2]; w = rng.integers(2, size=len(pair)); out['adjacent-pair audit'] = pair_interval(x[pair, w], x[pair, w + 1], hh[pair], rng)
    models_out, est = model_intervals(y, x[lab, 1], lab, hh, rng); out.update(models_out)
    est['record alone'] = pooled(y); est['adjacent-pair audit'] = phi(x[pair, w], x[pair, w + 1]) or float('nan'); out['point estimates'] = est
    return out


def score(reps, truth):
    reps = [r for r in reps if r is not None]; out = {}
    for k in reps[0]:
        if k == 'point estimates': continue
        a = [r[k] for r in reps]; ok = np.array([q is not None for q in a]); e = np.array([q if q is not None else (np.nan, np.nan) for q in a], float)
        hit = ok & (np.nan_to_num(e[:, 0], nan=9) - 1e-9 <= truth) & (truth <= np.nan_to_num(e[:, 1], nan=-9) + 1e-9)
        out[k] = dict(replicates=len(a), empty_rate=float((~ok).mean()), coverage=float(hit.mean()), width=float((e[ok, 1] - e[ok, 0]).mean()) if ok.any() else None,
                      positive=float((ok & (np.nan_to_num(e[:, 0], nan=-9) > 0)).mean()))
    for k in reps[0].get('point estimates', {}):
        v = np.array([r['point estimates'][k] for r in reps], float); v = v[~np.isnan(v)]
        out[k].update(median_estimate=float(np.median(v)), median_absolute_error=float(np.median(np.abs(v - truth))))
    return out


# ---------------------------------------------------------------- parallel jobs (pools are inherited by the workers)
POOLS = {}; LAWS = {}


def job_pool(a):
    name, labels, r, models = a; rng = np.random.default_rng([940001, r, labels, sorted(POOLS).index(name)]); x, y, hh = resample(POOLS[name], rng)
    return evaluate(x, y, hh, labels, rng, models) if labels <= len(x) else None


def job_stability(a):
    """Two-class model fitted from 4 and from 24 random starts on one resample: absolute difference of the estimates."""
    name, labels, r = a; rng = np.random.default_rng([960001, r, labels]); x, y, hh = resample(POOLS[name], rng); lab = rng.permutation(len(x))[:labels]
    cells = y @ POW; cells[lab] = 8 + 8 * x[lab, 1] + cells[lab]; total = np.bincount(cells, minlength=24).astype(float); un, au = total[:8], total[8:].reshape(2, 8)
    start = lambda: np.array([rng.uniform(.1, .9)] + [rng.uniform(.05, .95) for _ in range(6)] + [.01])
    return abs(class_corr(class_fit(un, au, [start() for _ in range(4)])) - class_corr(class_fit(un, au, [start() for _ in range(24)])))


CONTROL = {}


def job_control(a):
    """Positive control: windows drawn from a hidden Markov model fitted to the data set; every model and the neighbor finite-sample interval are applied."""
    name, n, labels, r = a; rng = np.random.default_rng([970001, r, sorted(CONTROL).index(name)]); i = rng.choice(16, size=n, p=CONTROL[name][0]); x1, y = i // 8, PATTERNS[i % 8]
    order = rng.permutation(n); aud, cal, lab = order[:labels // 2], order[labels // 2:labels], order[:labels]; eps = cp_upper(int(np.sum((x1[cal] == 0) & (y[cal, 1] == 1))), len(cal), .01)
    out, est = model_intervals(y, x1[lab], lab, np.arange(n), rng); out['finite-sample N'] = finite_sample(y, x1[aud], y[aud], eps, .04, 'N'); out['point estimates'] = est
    return out


def job_law(a):
    tag, n, labels, r = a; rng = np.random.default_rng([950001, r, tag]); i = rng.choice(NW, size=n, p=LAWS[tag]); return evaluate(WX[i], WY[i], np.arange(n), labels, rng)


def job_population(a):
    name, r = a; rng = np.random.default_rng([930001, r, sorted(POOLS).index(name)]); x, y, _ = resample(POOLS[name], rng); s = sharp(project_ls(window_law(x, y)))
    return dict(corr=pooled(x), corr_record=pooled(y), M=s['M']['width'], N=s['N']['width'], F=s['F']['width'], M_minus_N=s['M']['width'] - s['N']['width'], N_minus_F=s['N']['width'] - s['F']['width'],
                reduction_N_vs_M=1 - s['N']['width'] / s['M']['width'], N_lower=s['N']['lower'], N_upper=s['N']['upper'])


def summarize_pool(name):
    p = POOLS[name]; x, y = p['x'], p['y']; q = window_law(x, y); ls, tv = project_ls(q), project_tv(q)
    return dict(windows=len(x), participants=p['participants'], households=int(p['hh'].max() + 1), prevalence=float(x.mean()), record_prevalence=float(y.mean()), miss_rate=float(np.mean(y[x == 1] == 0)),
                false_positive_share=float(np.mean(x[y == 1] == 0)) if y.sum() else None, corr=pooled(x), corr_record=pooled(y), distance_to_stationary=float(.5 * np.abs(tv - q).sum()),
                population=sharp(ls), population_total_variation=sharp(tv))


def interval(values): return dict(lo=float(np.percentile(values, 2.5)), hi=float(np.percentile(values, 97.5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cohort', choices=sorted(LOADERS), help='data set'); ap.add_argument('--output', type=Path, required=True, help='folder for <data set>_cohort.json (an existing file is not overwritten)')
    ap.add_argument('--workers', type=int, default=8, help='parallel processes'); ap.add_argument('--replicates', type=int, default=1000, help='unit resamples per number of labels, and samples per extreme law')
    ap.add_argument('--sensitivity-replicates', type=int, default=300, help='unit resamples for the other window definitions')
    ap.add_argument('--positive-control', action='store_true', help='only run the positive control (windows drawn from hidden Markov models fitted to the data set) and write <data set>_control.json')
    ap.add_argument('--bootstrap', type=int, default=1000, help='unit resamples for the intervals of the population quantities (each evaluation uses its own %d inner bootstrap samples)' % BOOT)
    a = ap.parse_args(); out = a.output / (f'{a.cohort}_control.json' if a.positive_control else f'{a.cohort}_cohort.json')
    if out.exists(): ap.error(f'{out} exists; refusing to overwrite')
    missing = [str(DATA / f) for f in INPUTS[a.cohort] if not (DATA / f).is_file()]
    if missing: ap.error('input files not found: ' + ', '.join(missing) + '. The recorded data are not redistributed; see data/raw/recorded_data/DATA_SOURCE.md '
                         'for the sources, or set the environment variable RECORDED_DATA to the folder that holds them.')
    t0 = time.time(); data = LOADERS[a.cohort]()
    if a.positive_control:                                                # each single-chain model is fitted to the fully labelled data set before the workers start
        pool = build(data, CONFIGS[a.cohort]['all windows']); x, y = pool['x'], pool['y']; aud = np.zeros((2, 8)); np.add.at(aud, (x[:, 1], y @ POW), 1)
        for name, (sp, box) in HMM.items():
            extra = [.01 if hi < .5 else .5 for _, hi in box]; t = hmm_fit(name, np.zeros(8), aud, [np.r_[u, v, extra] for u, v in ((.2, .6), (.05, .9), (.4, .5))])
            cells = cell_probabilities(*chain(t[0], t[1]), *sp(t[2:])).ravel(); CONTROL[name] = (cells / cells.sum(), float(chain(t[0], t[1])[1]))
        with Pool(a.workers) as P:
            res = {name: dict(generating_correlation=CONTROL[name][1], windows=len(x), labels=400, replicates=300, results=score(P.map(job_control, [(name, len(x), 400, r) for r in range(300)], chunksize=2), CONTROL[name][1])) for name in list(CONTROL)}
        a.output.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(dict(cohort=a.cohort, positive_control=res), indent=1) + '\n'); print('wrote', out, round(time.time() - t0), flush=True); return
    POOLS.update({k: build(data, c) for k, c in CONFIGS[a.cohort].items()})
    res = dict(cohort=a.cohort, settings=dict(replicates=a.replicates, sensitivity_replicates=a.sensitivity_replicates, bootstrap=a.bootstrap, inner_bootstrap=BOOT, alpha=ALPHA), description=describe(data[0]),
               pools={k: summarize_pool(k) for k in POOLS}, methods=[], sensitivity=[])
    main_pool = 'all windows'; q = project_ls(window_law(POOLS[main_pool]['x'], POOLS[main_pool]['y'])); la, lb, ca, cb_, gap = equivalent_laws(q); LAWS.update({0: la, 1: lb}); n = len(POOLS[main_pool]['x'])
    with Pool(a.workers) as P:
        for k in (main_pool, 'one window per participant'):
            b = P.map(job_population, [(k, r) for r in range(a.bootstrap)], chunksize=10); res['pools'][k]['bootstrap'] = {c: interval([v[c] for v in b]) for c in b[0]}
        print('population', round(time.time() - t0), flush=True)
        for k, budgets in ((main_pool, (200, 400, 1000, 2000)), ('one window per participant', (200, 400))):
            for g in budgets:
                if g > .8 * len(POOLS[k]['x']): continue
                res['methods'].append(dict(pool=k, labels=g, truth=res['pools'][k]['corr'], results=score(P.map(job_pool, [(k, g, r, True) for r in range(a.replicates)], chunksize=2), res['pools'][k]['corr'])))
                print(k, g, round(time.time() - t0), flush=True)
        res['equivalent_laws'] = dict(corr_low=ca, corr_high=cb_, max_difference_of_observable_moments=gap, windows=n, labels=400, minimal_expected_width=(1 - 2 * ALPHA) * (cb_ - ca),
                                      low=score(P.map(job_law, [(0, n, 400, r) for r in range(a.replicates)], chunksize=2), ca), high=score(P.map(job_law, [(1, n, 400, r) for r in range(a.replicates)], chunksize=2), cb_))
        d = np.array(P.map(job_stability, [(main_pool, 400, r) for r in range(100)], chunksize=2))
        res['two_class_start_check'] = dict(replicates=len(d), few_starts=4, many_starts=24, median_difference=float(np.median(d)), share_within_0_02=float(np.mean(d <= .02)), largest_difference=float(d.max()))
        print('equivalent laws', round(time.time() - t0), flush=True)
        for k in POOLS:
            g = next((g for g in (400, 200) if g <= .8 * len(POOLS[k]['x'])), None)
            if k == main_pool or g is None: continue
            res['sensitivity'].append(dict(pool=k, labels=g, truth=res['pools'][k]['corr'], results=score(P.map(job_pool, [(k, g, r, False) for r in range(a.sensitivity_replicates)], chunksize=2), res['pools'][k]['corr'])))
        print('sensitivity', round(time.time() - t0), flush=True)
    a.output.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1) + '\n'); print('wrote', out, round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
