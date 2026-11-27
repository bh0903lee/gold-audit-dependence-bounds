"""The compressed six-variable confidence program and its external comparators (three-point windows only).

All procedures here use the same information: the observed Y triples and, from the audit cohort,
the recorded X_1 with its neighbors Y_0, Y_2.  Nothing about the simulator's Markov or detector
structure is supplied to any of them.

Five moments (manuscript eq. (moments)):  m=E X_1, a=E[Y_0 X_1], b=E[X_1 Y_2] from the audit cohort;
y=E Y_1, h=E[Y_0 Y_1] from the observation cohort.
"""
import itertools
import numpy as np
from scipy.optimize import linprog
from . import engine

NAMES = ('m', 'y', 'a', 'b', 'h')
OBS_PATTERNS = np.array(list(itertools.product(range(2), repeat=3)))        # index = 4*y0 + 2*y1 + y2
# Audit cells: {X1=0} and {X1=1} x {(Y0,Y2)} in the order (0,0),(0,1),(1,0),(1,1).
AUDIT_CELLS = 5


def closed_form(m, y, a, b, h):
    """Theorem 1 endpoints (theta_L, theta_U); works elementwise on arrays. No compatibility check."""
    lower = np.maximum(1 - 4*m + 4*a + 4*b - 4*h, 4*m - 3)
    upper = 1 - 4*y + 4*np.minimum(a, b)
    return lower, upper


def compatible(m, y, a, b, h, tol=1e-12):
    """Corollary 2: 0<=y<=m<=1, fixed entries of K nonnegative and l<=u. Elementwise."""
    lo = np.maximum(0., 2*m - 1 - a - b + h)
    hi = np.minimum(m - y - a + h, m - y - b + h)
    fixed = np.minimum.reduce([h, a - h, b - h, y - a, y - b, y, m - y, 1 - m])
    return (fixed >= -tol) & (lo <= hi + tol)


def six_variable_program(limits):
    """min/max of 1-4m+4a+4b-4h+4t over (m,y,a,b,h,t) with box limits on the five moments and K(t)>=0.

    limits: mapping name -> (low, high) for m,y,a,b,h.  Raises engine.InfeasibleConfidenceSet if the set is empty
    and engine.SolverFailure if the solver stops without a solution or an infeasibility certificate.
    """
    # Rows are -K_ij <= rhs, variables ordered (m, y, a, b, h, t).
    A = np.array([
        [2, 0, -1, -1, 1, -1],     # K00 = 1-2m+a+b-h+t >= 0
        [-1, 1, 1, 0, -1, 1],      # K01 = m-y-a+h-t   >= 0
        [-1, 1, 0, 1, -1, 1],      # K10 = m-y-b+h-t   >= 0
        [0, -1, 0, 1, 0, 0],       # K02 = y-b >= 0
        [0, -1, 1, 0, 0, 0],       # K20 = y-a >= 0
        [0, 0, -1, 0, 1, 0],       # K21 = a-h >= 0
        [0, 0, 0, -1, 1, 0],       # K12 = b-h >= 0
    ], float)
    rhs = np.array([1., 0, 0, 0, 0, 0, 0])
    # K22 = h >= 0 and K11 = t >= 0 are variable bounds.
    # Probabilities live in [0,1]; clip both ends so a limit that overshoots by rounding cannot empty a box.
    box = [(min(1., max(0., float(limits[k][0]))), max(0., min(1., float(limits[k][1])))) for k in NAMES] + [(0., None)]
    if any(lo > hi for lo, hi in box[:5]):
        raise engine.InfeasibleConfidenceSet('Confidence set infeasible: empty moment interval')
    c = np.array([-4., 0, 4, 4, -4, 4])
    out = []
    for sign in (1, -1):
        r = linprog(sign*c, A_ub=A, b_ub=rhs, bounds=box, method='highs')
        if r.success and np.max(A@r.x - rhs) > 1e-7:
            # Near-degenerate boxes can leave a presolve residual just above tolerance; solve again without presolve.
            r = linprog(sign*c, A_ub=A, b_ub=rhs, bounds=box, method='highs', options=dict(presolve=False))
        if not r.success:
            raise engine.lp_failure(r)
        if np.max(A@r.x - rhs) > 1e-7:
            raise AssertionError('LP residual')
        out.append(float(1 + c@r.x))
    return out


def five_moment_samples(y, audit_x, audit_y):
    """0/1 sample columns whose means estimate (m,y,a,b,h)."""
    t = engine.AUDIT_T
    audit_x = np.asarray(audit_x); audit_y = np.asarray(audit_y); y = np.asarray(y)
    return dict(m=audit_x.astype(float), y=y[:, t].astype(float),
                a=(audit_y[:, t-1]*audit_x).astype(float), b=(audit_x*audit_y[:, t+1]).astype(float),
                h=(y[:, t-1]*y[:, t]).astype(float))


def six_variable_limits(y, audit_x, audit_y, alpha=.05, interval='exact_binomial'):
    """Five simultaneous two-sided limits, each at error alpha/5 (union bound over exactly five features)."""
    delta = alpha/len(NAMES)
    return {k: engine.confidence_limits(v, delta, interval) for k, v in five_moment_samples(y, audit_x, audit_y).items()}


def six_variable_bounds(y, audit_x, audit_y, alpha=.05, interval='exact_binomial'):
    if np.asarray(y).shape[1] != 3:
        raise ValueError('The six-variable program is defined for three-point windows')
    return six_variable_program(six_variable_limits(y, audit_x, audit_y, alpha, interval))


def population_interval(m, y, a, b, h):
    """Six-variable program at exact moments (degenerate boxes): the identified interval."""
    return six_variable_program({k: (v, v) for k, v in zip(NAMES, (m, y, a, b, h))})


# ---- sufficient statistics --------------------------------------------------------------------------
def obs_counts(y):
    y = np.asarray(y)
    return np.bincount(4*y[:, 0] + 2*y[:, 1] + y[:, 2], minlength=8)


def audit_counts16(audit_x, audit_y):
    """Counts over (recorded X1, Y0, Y1, Y2); index = 8*x + 4*y0 + 2*y1 + y2."""
    audit_x = np.asarray(audit_x); audit_y = np.asarray(audit_y)
    return np.bincount(8*audit_x + 4*audit_y[:, 0] + 2*audit_y[:, 1] + audit_y[:, 2], minlength=16)


def audit_counts5(audit_x, audit_y):
    audit_x = np.asarray(audit_x); audit_y = np.asarray(audit_y)
    return np.bincount(audit_x*(1 + 2*audit_y[:, 0] + audit_y[:, 2]), minlength=AUDIT_CELLS)


def moments_from_cells(p8, q5):
    """Five moments from observed-pattern probabilities (..., 8) and audit-cell probabilities (..., 5)."""
    p8 = np.asarray(p8, float); q5 = np.asarray(q5, float)
    y = p8[..., [2, 3, 6, 7]].sum(-1); h = p8[..., [6, 7]].sum(-1)
    m = 1 - q5[..., 0]; a = q5[..., 3] + q5[..., 4]; b = q5[..., 2] + q5[..., 4]
    return m, y, a, b, h


# ---- Bootstrap comparator: plug-in endpoints with a percentile bootstrap (no finite-sample guarantee) --------------------
def bootstrap_interval(oc, ac, alpha, rng, reps=2000):
    oc = np.asarray(oc); ac = np.asarray(ac); N = oc.sum(); B = ac.sum()
    p = rng.multinomial(N, oc/N, size=reps)/N
    q = rng.multinomial(B, ac/B, size=reps)/B
    lower, upper = closed_form(*moments_from_cells(p, q))
    lo = max(-1., float(np.quantile(lower, alpha/2))); hi = min(1., float(np.quantile(upper, 1 - alpha/2)))
    return lo, hi, dict(empty_interval=bool(lo > hi))


# ---- Dirichlet comparator: Dirichlet(1/2) posterior credible set (no frequentist guarantee) ------------------------------
def dirichlet_interval(oc, ac, alpha, rng, draws=4000, prior=.5, minimum=100):
    p = rng.dirichlet(np.asarray(oc) + prior, size=draws)
    q = rng.dirichlet(np.asarray(ac) + prior, size=draws)
    mom = moments_from_cells(p, q); keep = compatible(*mom)
    info = dict(discard_fraction=float(1 - keep.mean()), fallback=bool(keep.sum() < minimum))
    if info['fallback']:
        return -1., 1., info
    lower, upper = closed_form(*[v[keep] for v in mom])
    lo = max(-1., float(np.quantile(lower, alpha/2))); hi = min(1., float(np.quantile(upper, 1 - alpha/2)))
    info['empty_interval'] = bool(lo > hi)
    return lo, hi, info


# ---- L1-ball comparator: two Weissman L1 balls projected through the word LP (finite-sample guarantee) -----------------
def l1_radius(k, n, delta):
    """Weissman et al. (2003): P(||p_hat-p||_1 >= eps) <= (2^k-2) exp(-n eps^2/2)."""
    return float(np.sqrt(2*np.log((2**k - 2)/delta)/n))


def l1_ball_bounds(oc, ac, alpha=.05):
    oc = np.asarray(oc, float); ac = np.asarray(ac, float); N = oc.sum(); B = ac.sum()
    words, X, Y, eq, beq, target = engine._lp_setup(3); n = len(words); t = engine.AUDIT_T
    obs_f = [np.all(Y == o, axis=1).astype(float) for o in OBS_PATTERNS]
    aud_f = [1 - X[:, t]] + [X[:, t]*(Y[:, t-1] == r)*(Y[:, t+1] == s) for r in (0, 1) for s in (0, 1)]
    feats = obs_f + aud_f; centre = np.r_[oc/N, ac/B]; k = len(feats)      # 8 + 5 auxiliary deviations
    A = []; rhs = []
    for j, (f, c) in enumerate(zip(feats, centre)):
        e = np.zeros(k); e[j] = 1
        A.append(np.r_[f, -e]); rhs.append(c)       #  f.p - e_j <= c
        A.append(np.r_[-f, -e]); rhs.append(-c)     # -f.p - e_j <= -c
    A.append(np.r_[np.zeros(n), np.ones(8), np.zeros(5)]); rhs.append(l1_radius(8, N, alpha/2))
    A.append(np.r_[np.zeros(n), np.zeros(8), np.ones(5)]); rhs.append(l1_radius(5, B, alpha/2))
    A = np.array(A); rhs = np.array(rhs)
    Aeq = np.hstack([np.array(eq), np.zeros((len(eq), k))]); out = []
    for sign in (1, -1):
        r = linprog(sign*np.r_[target, np.zeros(k)], A_ub=A, b_ub=rhs, A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs')
        if not r.success:
            raise engine.lp_failure(r)
        if np.max(A@r.x - rhs) > 1e-7 or np.max(abs(Aeq@r.x - beq)) > 1e-7:
            raise AssertionError('LP residual')
        out.append(float(target@r.x[:n]))
    return out
