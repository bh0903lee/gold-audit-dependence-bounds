"""Assumption-violation study: contaminated simulators, their exact population laws and the population breakdown bound.

A *condition* is (axis, level, stay).  It fixes a generator for the observation cohort, a generator
for the audit cohort, an audit recording channel and an audit selection rule.  At level 0 every
violation axis consumes the random stream exactly like the baseline `engine.sample_episodes`.

Violation axes (the codes are the identifiers used in configs and result rows):
  V1  two-sided error (false positives, rate phi)          V4a random audit recording error
  V2a audit-cohort drift (stay or detector)                V4b audit error anchored on Y_1
  V2b within-window detector non-stationarity              V5  audit selection depending on Y_1
  V3a second-order latent chain   (inside the model class) V6  windows cut from one dependent series
  V3b history-dependent detector  (inside the model class)
"""
import itertools
import numpy as np
from . import engine, sixvar
from .common import stationary

DETECT = [.95, .2]
L = 3
DETECT_X2 = [[.95, .70], [.35, .05]]      # P(D_t=1 | X_{t-1}, X_{t-2})
DETECT_DPREV = [[.80, .99], [.05, .50]]   # P(D_t=1 | X_{t-1}, D_{t-1})

LEVELS = {
    'V1': [.002, .005, .01, .02, .05, .10],
    'V2a': ['stay+0.05', 'stay+0.1', 'stay+0.2', 'det+0.05', 'det+0.1'],
    'V2b': [.05, .10, .20],
    'V3a': [.25, .5, .75],
    'V3b': ['x2', 'dprev'],
    'V4a': [.01, .02, .05],
    'V4b': [.05, .10, .20],
    'V5': [2, 4],
    'V6': [1, 3, 30],
}
IN_MODEL = {'baseline', 'V3a', 'V3b'}     # axes that do not violate the theorem's assumptions


def markov1(stay, detect=DETECT, kappa=0., fp=0.):
    return dict(kind='markov1', stay=float(stay), detect=[float(detect[0]), float(detect[1])], kappa=float(kappa), fp=float(fp))


def make_condition(axis, level, stay):
    """Generators and audit channel for one condition. `level` is ignored for the baseline."""
    base = markov1(stay)
    c = dict(axis=axis, level=level, stay=float(stay), obs=base, aud=base, flip=0., anchor=0., rho=1., step=None)
    if axis == 'baseline':
        c['level'] = 0
    elif axis == 'V1':
        c['obs'] = c['aud'] = markov1(stay, fp=level)
    elif axis == 'V2a':
        if isinstance(level, str):
            kind, value = level.split('+'); value = float(value)
        else:
            kind, value = 'stay', float(level)
        if kind == 'stay':
            c['aud'] = markov1(min(1., stay + value))
        elif kind == 'det':
            c['aud'] = markov1(stay, detect=[DETECT[0] - value, DETECT[1] + value])
        else:
            raise ValueError('Unknown drift kind')
    elif axis == 'V2b':
        c['obs'] = c['aud'] = markov1(stay, kappa=level)
    elif axis == 'V3a':
        g = float(level)
        c['obs'] = c['aud'] = dict(kind='markov2', p_ss=stay + g*(1 - stay), p_cs=stay - g*stay, detect=list(DETECT))
    elif axis == 'V3b':
        if level not in ('x2', 'dprev'):
            raise ValueError('Unknown detector variant')
        c['obs'] = c['aud'] = dict(kind=level, stay=float(stay), table=DETECT_X2 if level == 'x2' else DETECT_DPREV)
    elif axis == 'V4a':
        c['flip'] = float(level)
    elif axis == 'V4b':
        c['anchor'] = float(level)
    elif axis == 'V5':
        c['rho'] = float(level)
    elif axis == 'V6':
        c['step'] = int(level)
    else:
        raise ValueError('Unknown axis ' + str(axis))
    return c


def grid(axes):
    """Baseline first, then every prespecified nonzero level of the requested axes."""
    return [('baseline', 0)] + [(a, lv) for a in axes for lv in LEVELS[a]]


# ---- simulators --------------------------------------------------------------------------------------
def _dprev_chain(stay, table):
    """Transition matrix of S_t=(X_t,D_t), index 2*x+d; D_t depends on (X_{t-1},D_{t-1}) only."""
    P = np.zeros((4, 4))
    for x, d, x2, d2 in itertools.product(range(2), repeat=4):
        pd = table[x][d]
        P[2*x + d, 2*x2 + d2] = (stay if x2 == x else 1 - stay)*(pd if d2 else 1 - pd)
    return P


def sample_latent(rng, N, L, gen):
    """True latent X and observed Y, both (N,L). The markov1 kind with kappa=fp=0 is the baseline sampler."""
    kind = gen['kind']
    x = np.zeros((N, L), int); y = np.zeros((N, L), int)
    if kind == 'markov1':
        stay = gen['stay']; d0, d1 = gen['detect']
        x[:, 0] = rng.integers(2, size=N)
        prev = rng.integers(2, size=N)               # same (discarded) draw as engine.sample_episodes
        prev = np.where(rng.random(N) < stay, x[:, 0], 1 - x[:, 0])
        for t in range(L):
            if t:
                x[:, t] = np.where(rng.random(N) < stay, x[:, t-1], 1 - x[:, t-1]); prev = x[:, t-1]
            y[:, t] = x[:, t]*(rng.random(N) < np.asarray([d0 - gen['kappa']*t, d1])[prev])
        if gen['fp'] > 0:                            # extra draws only when the violation is active
            y = y + (1 - x)*(rng.random((N, L)) < gen['fp'])
        return x, y
    if kind == 'markov2':
        pss, pcs = gen['p_ss'], gen['p_cs']; q = pcs/(1 - pss + pcs)
        x2 = rng.integers(2, size=N); x1 = np.where(rng.random(N) < q, x2, 1 - x2)
        for t in range(L):
            ps = np.where(x1 == x2, pss, pcs)
            x[:, t] = np.where(rng.random(N) < ps, x1, 1 - x1)
            y[:, t] = x[:, t]*(rng.random(N) < np.asarray(gen['detect'])[x1])
            x2, x1 = x1, x[:, t]
        return x, y
    if kind == 'x2':
        stay = gen['stay']; table = np.asarray(gen['table'])
        x2 = rng.integers(2, size=N); x1 = np.where(rng.random(N) < stay, x2, 1 - x2)
        for t in range(L):
            x[:, t] = np.where(rng.random(N) < stay, x1, 1 - x1)
            y[:, t] = x[:, t]*(rng.random(N) < table[x1, x2])
            x2, x1 = x1, x[:, t]
        return x, y
    if kind == 'dprev':
        stay = gen['stay']; table = np.asarray(gen['table'])
        s = rng.choice(4, size=N, p=stationary(_dprev_chain(stay, gen['table']))); x1 = s//2; d1 = s % 2
        for t in range(L):
            x[:, t] = np.where(rng.random(N) < stay, x1, 1 - x1)
            d = (rng.random(N) < table[x1, d1]).astype(int)
            y[:, t] = x[:, t]*d
            x1, d1 = x[:, t], d
        return x, y
    raise ValueError('Unknown generator kind')


def sample_series_windows(rng, n, L, stay, detect, step):
    """V6: n length-L windows with spacing `step` cut from ONE stationary baseline series."""
    T = (n - 1)*step + L
    x0 = int(rng.integers(2)); prev0 = x0 if rng.random() < stay else 1 - x0
    switch = (rng.random(T - 1) >= stay).astype(np.int64)
    x = np.r_[x0, (x0 + np.cumsum(switch)) % 2]
    prev = np.r_[prev0, x[:-1]]
    y = x*(rng.random(T) < np.asarray(detect)[prev])
    idx = np.arange(n)[:, None]*step + np.arange(L)[None, :]
    return x[idx], y[idx]


def sample_selected(rng, B, L, gen, rho):
    """V5: exactly B audit episodes from the law reweighted by w(Y_1)=rho if Y_1=1 else 1 (rejection sampling)."""
    if rho == 1:
        return sample_latent(rng, B, L, gen)
    if rho < 1:
        raise ValueError('rho must be >= 1')
    xs = []; ys = []; have = 0
    while have < B:
        n = int(2*rho*(B - have)) + 64
        x, y = sample_latent(rng, n, L, gen)
        keep = (y[:, engine.AUDIT_T] == 1) | (rng.random(n) < 1/rho)
        xs.append(x[keep]); ys.append(y[keep]); have += int(keep.sum())
    return np.concatenate(xs)[:B], np.concatenate(ys)[:B]


def record_audit(rng, x1, y1, flip, anchor):
    """Recorded audit label. V4a: flipped w.p. `flip`. V4b: a true 1 with Y_1=0 is recorded 0 w.p. `anchor`."""
    rec = np.array(x1, int)
    if flip > 0:
        rec = np.where(rng.random(len(rec)) < flip, 1 - rec, rec)
    if anchor > 0:
        rec = np.where((rng.random(len(rec)) < anchor) & (rec == 1) & (np.asarray(y1) == 0), 0, rec)
    return rec


def draw_condition(rng, cond, N, budgets, L=L):
    """Observation Y (N,L) and, per budget, (recorded X_1, audit Y, true X_1). Same call order as engine.run."""
    t = engine.AUDIT_T
    if cond['step'] is not None:
        g = cond['obs']; _, y = sample_series_windows(rng, N, L, g['stay'], g['detect'], cond['step'])
    else:
        _, y = sample_latent(rng, N, L, cond['obs'])
    audits = {}
    for B in budgets:
        if cond['step'] is not None:
            g = cond['aud']; x, ay = sample_series_windows(rng, B, L, g['stay'], g['detect'], cond['step'])
        else:
            x, ay = sample_selected(rng, B, L, cond['aud'], cond['rho'])
        audits[B] = (record_audit(rng, x[:, t], ay[:, t], cond['flip'], cond['anchor']), ay, x[:, t])
    return y, audits


# ---- exact population laws ---------------------------------------------------------------------------
def joint_law(gen, L=L):
    """Exact law of (X_0..X_{L-1}, Y_0..Y_{L-1}) as an array of shape (2,)*2L (X axes first)."""
    J = np.zeros((2,)*(2*L)); kind = gen['kind']

    def emit(px, xs, p1):
        # p1[t] = P(Y_t=1 | latent history); the Y_t are conditionally independent given that history.
        for ys in itertools.product(range(2), repeat=L):
            pr = px
            for t in range(L):
                pr *= p1[t] if ys[t] else 1 - p1[t]
            J[tuple(xs) + ys] += pr

    if kind == 'markov1':
        s = gen['stay']
        for prev in range(2):
            for xs in itertools.product(range(2), repeat=L):
                px = .5; hist = [prev] + list(xs)
                for t in range(L):
                    px *= s if hist[t+1] == hist[t] else 1 - s
                emit(px, xs, [(([gen['detect'][0] - gen['kappa']*t, gen['detect'][1]][hist[t]]) if xs[t] else gen['fp']) for t in range(L)])
    elif kind in ('markov2', 'x2'):
        for xm2, xm1 in itertools.product(range(2), repeat=2):
            if kind == 'markov2':
                q = gen['p_cs']/(1 - gen['p_ss'] + gen['p_cs']); p0 = .5*(q if xm1 == xm2 else 1 - q)
            else:
                p0 = .5*(gen['stay'] if xm1 == xm2 else 1 - gen['stay'])
            for xs in itertools.product(range(2), repeat=L):
                hist = [xm2, xm1] + list(xs); px = p0; p1 = []
                for t in range(L):
                    a, b, c = hist[t], hist[t+1], hist[t+2]
                    if kind == 'markov2':
                        ps = gen['p_ss'] if a == b else gen['p_cs']; det = gen['detect'][b]
                    else:
                        ps = gen['stay']; det = gen['table'][b][a]
                    px *= ps if c == b else 1 - ps; p1.append(det if c else 0.)
                emit(px, xs, p1)
    elif kind == 'dprev':
        s = gen['stay']; pi = stationary(_dprev_chain(s, gen['table']))
        for xm1, dm1 in itertools.product(range(2), repeat=2):
            for xs in itertools.product(range(2), repeat=L):
                for ds in itertools.product(range(2), repeat=L):
                    pr = pi[2*xm1 + dm1]; xp, dp = xm1, dm1
                    for t in range(L):
                        pd = gen['table'][xp][dp]
                        pr *= (s if xs[t] == xp else 1 - s)*(pd if ds[t] else 1 - pd); xp, dp = xs[t], ds[t]
                    J[tuple(xs) + tuple(x*d for x, d in zip(xs, ds))] += pr
    else:
        raise ValueError('Unknown generator kind')
    return J


def analyst_law(cond):
    """What the analyst's samples converge to, plus the truth for the observation-cohort population.

    Returns truth, obs8 (index 4y0+2y1+y2), audit16 (index 8*recorded x1+4y0+2y1+y2), the five moments
    the six-variable program estimates, and the per-audit probability of the T1 event {recorded X_1=0, Y_1=1}.
    """
    Jo = joint_law(cond['obs']); Ja = joint_law(cond['aud'])
    spin = np.array([-1., 1.])
    truth = float(np.einsum('abcdef,a,b->', Jo, spin, spin))
    truth_12 = float(np.einsum('abcdef,b,c->', Jo, spin, spin))
    obs = Jo.sum(axis=(0, 1, 2))                                  # (y0,y1,y2)
    W = Ja.sum(axis=(0, 2))                                       # (x1,y0,y1,y2)
    W = W*np.array([1., cond['rho']])[None, None, :, None]; W = W/W.sum()
    A = np.zeros_like(W)
    A[0] = (1 - cond['flip'])*W[0] + cond['flip']*W[1]
    A[1] = (1 - cond['flip'])*W[1] + cond['flip']*W[0]
    moved = cond['anchor']*A[1, :, 0, :].copy()                   # recorded 1 with Y_1=0 -> recorded 0
    A[1, :, 0, :] -= moved; A[0, :, 0, :] += moved
    m = float(A[1].sum()); a = float(A[1, 1].sum()); b = float(A[1, :, :, 1].sum())
    y = float(obs[:, 1].sum()); h = float(obs[1, 1].sum())
    return dict(truth=truth, truth_pair12=truth_12, obs8=obs.ravel(), audit16=A.ravel(), m=m, y=y, a=a, b=b, h=h,
                t1_probability=float(A[0, :, 1, :].sum()))


def breakdown(cond, word_lp=True):
    """Population breakdown d = max(theta_L* - theta, theta - theta_U*, 0) from contaminated moments."""
    law = analyst_law(cond); mom = [law[k] for k in sixvar.NAMES]
    lo, hi = (float(v) for v in sixvar.closed_form(*mom))
    out = dict(truth=law['truth'], truth_pair12=law['truth_pair12'], **{k: law[k] for k in sixvar.NAMES},
               theta_low=lo, theta_high=hi, d=max(lo - law['truth'], law['truth'] - hi, 0.),
               compatible=bool(sixvar.compatible(*mom, tol=1e-12)), t1_probability=law['t1_probability'])
    if word_lp:
        # The neighbor word LP at the same contaminated population moments (all 8 patterns + m,a,b).
        words, X, Y, eq, beq, target = engine._lp_setup(L); feats = engine.moment_features(X, Y, L, 'neighbor'); tiny = 1e-9
        obs = law['obs8']; aud = [law['m'], law['a'], law['b']]
        try:
            wl, wh = engine.solve_bounds(L, [max(0, v - tiny) for v in obs], [min(1, v + tiny) for v in obs], feats,
                                         [max(0, v - tiny) for v in aud], [min(1, v + tiny) for v in aud])
            out.update(word_low=wl, word_high=wh, word_feasible=True, word_d=max(wl - law['truth'], law['truth'] - wh, 0.))
        except engine.InfeasibleConfidenceSet:
            out.update(word_low=None, word_high=None, word_feasible=False, word_d=None)
    return out
