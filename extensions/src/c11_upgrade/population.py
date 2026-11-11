"""Stationary word/pair LPs and independently checkable exact certificates.

The Markov generators are example laws only; their Markov restrictions are not imposed as LP constraints.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog

X = np.array([0, 1, 1, 0], dtype=int)
Y = np.array([0, 0, 1, 1], dtype=int)
INFO = {"N": ("m", "a", "b"), "N6": ("m", "a", "b", "j"),
        "N7L": ("m", "a", "b", "j", "cL"),
        "N7R": ("m", "a", "b", "j", "cR"),
        "N8": ("m", "a", "b", "j", "cL", "cR")}


@lru_cache(None)
def features(length=3, states=4):
    if length not in (3, 4) or states not in (3, 4):
        raise ValueError("Only L=3/4 and three/four-state LPs are supported")
    words = np.array(list(product(range(states), repeat=length)))
    x, y = X[words], Y[words]
    obs_id = y @ (2 ** np.arange(length - 1, -1, -1))
    audit_id = x[:, 1] * 2**length + obs_id
    rows = {"m": x[:, 1], "y": y[:, 1], "a": y[:, 0]*x[:, 1],
            "b": x[:, 1]*y[:, 2], "h": y[:, 0]*y[:, 1],
            "j": x[:, 1]*y[:, 1], "cL": y[:, 0]*x[:, 1]*y[:, 1],
            "cR": x[:, 1]*y[:, 1]*y[:, 2], "fp": (1-x[:, 1])*y[:, 1]}
    for k in range(2**length):
        rows[f"o{k}"] = (obs_id == k).astype(int)
    for k in range(2**(length+1)):
        rows[f"u{k}"] = (audit_id == k).astype(int)
    prefixes = np.array(list(product(range(states), repeat=length-1)))
    flow = np.array([np.all(words[:, :-1] == w, axis=1).astype(int)
                     - np.all(words[:, 1:] == w, axis=1) for w in prefixes])
    # Keep integral coefficients: stationarity makes the first-pair objective equal
    # to the objective averaged over adjacent pairs, checked separately in tests.
    c = (2*x[:, 0]-1)*(2*x[:, 1]-1)
    return words, rows, flow, c


@dataclass
class Problem:
    c: np.ndarray
    ae: np.ndarray
    be: np.ndarray
    au: np.ndarray
    bu: np.ndarray
    labels: tuple


def word_problem(limits, epsilon, info="N", length=3, states=4):
    _, rows, flow, c = features(length, states)
    if not 0 <= float(epsilon) <= 1:
        raise ValueError("epsilon is a joint probability in [0,1]")
    labels = [f"o{k}" for k in range(2**length)]
    labels += ([f"u{k}" for k in range(2**(length+1))]
               if info == "F" else list(INFO[info]))
    ae, be = [np.ones(len(c), int), *flow], [1, *([0]*len(flow))]
    au, bu = [rows["fp"]], [epsilon]
    for name in labels:
        lo, hi = limits[name]
        if float(lo) > float(hi):
            raise ValueError(f"Reversed limits: {name}")
        if lo == hi:
            ae.append(rows[name]); be.append(lo)
        else:
            au.extend([rows[name], -rows[name]]); bu.extend([hi, -lo])
    return Problem(np.asarray(c), np.asarray(ae), np.asarray(be, dtype=object),
                   np.asarray(au), np.asarray(bu, dtype=object), tuple(labels))


def lp_interval(problem, return_solutions=False):
    out = {"status": "ok", "lower": None, "upper": None, "residual": 0.0}
    sols = []
    for sign in (1, -1):
        r = None
        for presolve in (True, False):
            r = linprog(sign*problem.c, A_ub=np.asarray(problem.au, float),
                        b_ub=np.asarray(problem.bu, float), A_eq=np.asarray(problem.ae, float),
                        b_eq=np.asarray(problem.be, float), bounds=(0, None), method="highs",
                        options={"presolve": presolve, "primal_feasibility_tolerance": 1e-9,
                                 "dual_feasibility_tolerance": 1e-9})
            if r.status == 2:
                # Infeasibility with presolve on is re-checked with presolve off; the interval is empty only if both solves are infeasible.
                if presolve:
                    continue
                out["status"] = "empty"; out["lower"] = out["upper"] = None; break
            if r.success:
                res = max(float(np.max(np.abs(problem.ae@r.x-np.asarray(problem.be,float)))),
                          float(np.max(problem.au@r.x-np.asarray(problem.bu,float))),
                          float(-np.min(r.x)))
                if res <= 1e-7:
                    break
        if out["status"] == "empty":
            break
        if not r.success or res > 1e-7:
            out["status"] = "solver_failure"; out["message"] = r.message; break
        out["residual"] = max(out["residual"], res)
        out["lower" if sign == 1 else "upper"] = float(problem.c @ r.x)
        sols.append(r)
    if out["status"] == "ok" and (out["lower"] > out["upper"]+1e-8
                                   or out["lower"] < -1-1e-8 or out["upper"] > 1+1e-8):
        out["status"] = "solver_failure"; out["message"] = "Invalid endpoint ordering/domain"
    if return_solutions:
        out["solutions"] = sols
    return out


def law_limits(law, length=3, states=4):
    _, rows, _, _ = features(length, states)
    p = np.asarray(law, dtype=object).reshape(-1)
    return {name: (row@p, row@p) for name, row in rows.items() if name != "fp"}


def fp_mass(law, length=3):
    return features(length)[1]["fp"] @ np.asarray(law).reshape(-1)


def full_h(law):
    words, _, _, _ = features(3)
    h = np.zeros((2, 4, 2), dtype=object)
    for w, p in zip(words, np.asarray(law, dtype=object).reshape(-1)):
        h[Y[w[0]], w[1], Y[w[2]]] += p
    return h


def pair_problem(h):
    h = np.asarray(h, dtype=object)
    if h.shape != (2,4,2):
        raise ValueError("Three-point H required")
    hl, hr = h.sum(axis=2), h.sum(axis=0)
    ae, be = [np.ones(16, int)], [1]
    for z in range(4):
        flow = np.zeros((4,4), int); flow[z,:] += 1; flow[:,z] -= 1
        ae.append(flow.ravel()); be.append(0)
        for r in range(2):
            left = np.zeros((4,4), int); left[Y == r, z] = 1
            right = np.zeros((4,4), int); right[z,Y == r] = 1
            ae.extend([left.ravel(), right.ravel()]); be.extend([hl[r,z],hr[z,r]])
    c = np.outer(2*X-1, 2*X-1).ravel()
    return Problem(c, np.asarray(ae), np.asarray(be,dtype=object),
                   np.zeros((1,16),int), np.array([0],dtype=object), ("M8",))


def pair_from_moments(moments):
    """Build the pair LP from the eight central moments (m, y, a, b, h, j, cL, cR) only; the full audit law H is not needed."""
    vals=[]
    for name in ("m","y","a","b","h","j","cL","cR"):
        value=moments[name]
        if isinstance(value,(tuple,list)):
            if value[0]!=value[1]: raise ValueError("The pair LP requires exact (point) moments")
            value=value[0]
        if not np.isfinite(float(value)) or not 0<=float(value)<=1: raise ValueError("Invalid moment probability")
        vals.append(rational(value))
    m,y,a,b,h,j,cl,cr=vals
    q=np.array([1-m-y+j,m-j,j,y-j],dtype=object)
    left=np.array([y-a-h+cl,a-cl,cl,h-cl],dtype=object)
    right=np.array([y-b-h+cr,b-cr,cr,h-cr],dtype=object)
    if any(v<0 for v in [*q,*left,*(q-left),*right,*(q-right)]):
        raise ValueError("Moments are incompatible with any pair law")
    hl=np.stack([q-left,left]); hr=np.stack([q-right,right],axis=1)
    # This artificial H is just a carrier for the two known marginals.
    # The true full audit law is neither needed nor reconstructed here.
    carrier=np.zeros((2,4,2),dtype=object)
    for r,z,s in product(range(2),range(4),range(2)):
        if q[z]!=0: carrier[r,z,s]=hl[r,z]*hr[z,s]/q[z]
    return pair_problem(carrier)


def glue_triple(k, h):
    """Exact object arithmetic is retained for Fraction inputs; zero fibers map to 0."""
    k, h = np.asarray(k,dtype=object).reshape(4,4), np.asarray(h,dtype=object)
    hl, hr = h.sum(axis=2), h.sum(axis=0)
    t = np.zeros((4,4,4),dtype=object)
    for i,z,j in product(range(4), repeat=3):
        a,b = hl[Y[i],z],hr[z,Y[j]]
        if a != 0 and b != 0:
            t[i,z,j] = h[Y[i],z,Y[j]]*(k[i,z]/a)*(k[z,j]/b)
        elif h[Y[i],z,Y[j]] != 0:
            raise ValueError("Incompatible zero denominator")
    return t


def check_gluing(k, h, t, exact=False):
    """Certificate checker: separate summations, not the construction formula."""
    k,h,t = np.asarray(k,dtype=object).reshape(4,4),np.asarray(h,dtype=object),np.asarray(t,dtype=object)
    same = lambda a,b: bool(np.all(a == b)) if exact else bool(np.allclose(np.asarray(a,float),np.asarray(b,float),atol=1e-9,rtol=0))
    hh = np.zeros((2,4,2),dtype=object)
    for i,z,j in product(range(4), repeat=3): hh[Y[i],z,Y[j]] += t[i,z,j]
    spin = 2*X-1
    target_k = sum(k[i,j]*int(spin[i]*spin[j]) for i,j in product(range(4),repeat=2))
    target_t = sum(t[i,z,j]*Q(int(spin[i]*spin[z]+spin[z]*spin[j]),2)
                   for i,z,j in product(range(4),repeat=3))
    checks = {"nonnegative": bool(np.all(t >= 0)), "normalization": same(t.sum(),1),
              "left_pair": same(t.sum(axis=2),k), "right_pair": same(t.sum(axis=0),k),
              "full_h": same(hh,h), "target": same(target_t,target_k),
              "stationary_pair": same(k.sum(axis=0),k.sum(axis=1))}
    # The pair-state transition T(i,z,j)/K(i,z) has invariant mass K.
    incoming = np.zeros((4,4),dtype=object)
    for i,z,j in product(range(4),repeat=3):
        if k[i,z] != 0: incoming[z,j] += k[i,z]*(t[i,z,j]/k[i,z])
    checks["pair_chain_invariance"] = same(incoming,k)
    return checks


def rational(value):
    return value if isinstance(value,Q) else Q(str(value))


def exact_certificate(problem, result, sign=1):
    """Reconstruct sparse primal/dual solutions and verify exact LP optimality.

    No float gap is promoted to proof. A failed reconstruction returns verified=False.
    """
    import sympy as sp
    mat = lambda a: sp.Matrix([[sp.Rational(str(v)) for v in row] for row in a])
    vec = lambda a: sp.Matrix([sp.Rational(str(v)) for v in a])
    ae,au,be,bu,c = mat(problem.ae),mat(problem.au),vec(problem.be),vec(problem.bu),vec(sign*problem.c)
    support = np.flatnonzero(result.x > 1e-9).tolist()
    active = np.flatnonzero(np.abs(np.asarray(problem.au,float)@result.x-np.asarray(problem.bu,float)) < 1e-8).tolist()
    def solve(a,b):
        sol, pars = a.gauss_jordan_solve(b)
        return sol.subs({p:0 for p in pars})
    try:
        aa = ae.col_join(au[active,:]) if active else ae
        bb = be.col_join(bu[active,:]) if active else be
        values = solve(aa[:,support],bb)
        p = sp.zeros(len(c),1)
        for i,v in zip(support,values): p[i]=v
        # Reconstruct only nonzero numerical multipliers to avoid an arbitrary
        # free dual solution that might violate inequality multiplier signs.
        je = np.flatnonzero(np.abs(result.eqlin.marginals)>1e-9).tolist()
        ju = np.flatnonzero(np.abs(result.ineqlin.marginals)>1e-9).tolist()
        aa_dual = ae[je,:].T.row_join(au[ju,:].T)
        de=sp.Matrix([sp.Rational(str(Q(float(v)).limit_denominator(1000000))) for v in result.eqlin.marginals])
        du=sp.Matrix([sp.Rational(str(Q(float(v)).limit_denominator(1000000))) for v in result.ineqlin.marginals])
        dual_ok=(all(v<=0 for v in du) and all(v>=0 for v in c-ae.T*de-au.T*du)
                 and (c.T*p)[0]==(be.T*de+bu.T*du)[0])
        if not dual_ok:
            vals = solve(aa_dual[support,:],c[support,:])
            de,du = sp.zeros(ae.rows,1),sp.zeros(au.rows,1)
            for i,v in zip(je,vals[:len(je)]): de[i]=v
            for i,v in zip(ju,vals[len(je):]): du[i]=v
        checks = {"primal_nonnegative": all(v>=0 for v in p),
                  "equalities": ae*p == be, "inequalities": all(v<=0 for v in au*p-bu),
                  "dual_sign": all(v<=0 for v in du),
                  "dual_slack": all(v>=0 for v in c-ae.T*de-au.T*du),
                  "zero_duality_gap": (c.T*p)[0] == (be.T*de+bu.T*du)[0]}
        return {"verified": all(checks.values()), "checks": checks, "sign":sign,
                "objective": str((vec(problem.c).T*p)[0]),
                "primal": [str(v) for v in p], "dual_equality": [str(v) for v in de],
                "dual_inequality": [str(v) for v in du]}
    except (ValueError, ZeroDivisionError) as e:
        return {"verified": False, "reason": str(e), "sign":sign}


def periodic_law(seed, length=3, alphabet=4):
    rng=np.random.default_rng(seed)
    sizes=rng.integers(3,9,size=3)
    cycles=[rng.integers(0,alphabet,size=int(n)) for n in sizes]
    weights=rng.integers(1,6,size=3)
    law=np.zeros((4,)*length,dtype=object)
    for cycle,weight in zip(cycles,weights):
        for start in range(len(cycle)):
            w=tuple(cycle[(start+np.arange(length))%len(cycle)])
            law[w] += Q(int(weight),int(weights.sum())*len(cycle))
    return law


def markov_law(stay=.5, detect=(.95,.2), phi=0., length=3, prevalence=None, rho=.6):
    """Exact law of L consecutive (X, Y) states under the Markov latent/detector model (enumerates X[-1:L] and detector histories)."""
    probabilities=[stay,phi,rho,*detect]+([] if prevalence is None else [prevalence])
    if length not in (3,4) or len(detect)!=2 or any(not np.isfinite(float(v)) or not 0<=float(v)<=1 for v in probabilities):
        raise ValueError("Invalid length or generator probability")
    if prevalence is None:
        m=Q(1,2); s=rational(stay); trans=[[s,1-s],[1-s,s]]
    else:
        m=rational(prevalence); r=rational(rho)
        p01=m*(1-r); p10=(1-m)*(1-r); trans=[[1-p01,p01],[p10,1-p10]]
    detect=list(map(rational,detect)); phi=rational(phi)
    law=np.zeros((4,)*length,dtype=object)
    encode={(int(X[z]),int(Y[z])):z for z in range(4)}
    for hist in product(range(2),repeat=length+1):
        px=m if hist[0] else 1-m
        for i in range(length): px *= trans[hist[i]][hist[i+1]]
        for ys in product(range(2),repeat=length):
            mass=px
            for i,y in enumerate(ys):
                py=detect[hist[i]] if hist[i+1] else phi
                mass *= py if y else 1-py
            law[tuple(encode[hist[i+1],ys[i]] for i in range(length))] += mass
    return law


def closed_form(limits):
    m,y,a,b,h=(float(limits[k][0]) for k in ("m","y","a","b","h"))
    return max(1-4*m+4*a+4*b-4*h,4*m-3),1-4*y+4*min(a,b)
