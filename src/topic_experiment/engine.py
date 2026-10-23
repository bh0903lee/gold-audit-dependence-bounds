"""Finite-sample outer bounds from isolated audits and neighboring observations (word program).

Provides Hoeffding, empirical-Bernstein and exact Clopper-Pearson intervals; the marginal, neighbor and 'full'
audit moment sets (the full set is the joint law of the audited X_1 with every observed Y pattern); and the exact
population identified set, so each finite-sample width decomposes into the identified width plus concentration slack.
"""
from .common import AuditRows
import itertools
import numpy as np
from scipy.optimize import linprog

AUDIT_T=1  # the audited (revealed) latent position inside each episode

class InfeasibleConfidenceSet(RuntimeError):
    """The solver certified that the confidence-set constraints have no solution (an empty confidence set)."""

class SolverFailure(Exception):
    """The solver stopped without a solution or an infeasibility certificate (iteration limit, numerical
    difficulty, unboundedness). Never treated as an empty confidence set; the run fails with this message."""

def lp_failure(r):
    """Exception for an unsuccessful scipy.optimize.linprog result: status 2 (infeasible) or any other status."""
    if r.status==2: return InfeasibleConfidenceSet('Confidence set infeasible: '+r.message)
    return SolverFailure(f'LP solver failure (status {r.status}): {r.message}')

def sample_episodes(rng,N,L,stay,detect):
    x=np.zeros((N,L),int); x[:,0]=rng.integers(2,size=N)
    # Discarded draw, kept so that the random stream matches the archived runs; removing it changes every sample.
    prev=rng.integers(2,size=N)
    # Stationary reverse conditional for the unobserved predecessor of X0.
    prev=np.where(rng.random(N)<stay,x[:,0],1-x[:,0])
    y=np.zeros_like(x)
    for t in range(L):
        if t: x[:,t]=np.where(rng.random(N)<stay,x[:,t-1],1-x[:,t-1]); prev=x[:,t-1]
        y[:,t]=x[:,t]*(rng.random(N)<np.asarray(detect)[prev])
    return x,y

def _lp_setup(L):
    words=np.array(list(itertools.product(range(3),repeat=L)))
    X=(words>0).astype(float); Y=(words==2).astype(float); n=len(words)
    eq=[np.ones(n)]; beq=[1.]
    for w in itertools.product(range(3),repeat=L-1):
        eq.append(np.all(words[:,:-1]==w,axis=1).astype(float)-np.all(words[:,1:]==w,axis=1).astype(float)); beq.append(0.)
    target=np.mean((2*X[:,:-1]-1)*(2*X[:,1:]-1),axis=1)
    return words,X,Y,eq,beq,target

def moment_features(X,Y,L,moments):
    """Audit features f(X_1,Y) whose expectations the audit sample estimates."""
    t=AUDIT_T; feats=[X[:,t]]
    if moments=='neighbor':
        feats+=[X[:,t]*Y[:,t-1],X[:,t]*Y[:,t+1]]
    elif moments=='full':
        feats=[X[:,t]*np.all(Y==obs,axis=1) for obs in itertools.product(range(2),repeat=L)]
    elif moments!='marginal': raise ValueError('Unknown moment set')
    return feats

def audit_values(audit_x,audit_y,L,moments):
    """Sample realizations of the same features, one column per feature."""
    t=AUDIT_T; cols=[audit_x.astype(float)]
    if moments=='neighbor':
        cols+=[audit_x*audit_y[:,t-1],audit_x*audit_y[:,t+1]]
    elif moments=='full':
        cols=[audit_x*np.all(audit_y==obs,axis=1) for obs in itertools.product(range(2),repeat=L)]
    return [np.asarray(c,float) for c in cols]

def radius(values,delta,interval):
    """Two-sided confidence radius for the mean of i.i.d. values in [0,1]."""
    N=len(values)
    if interval=='hoeffding': return float(np.sqrt(np.log(2/delta)/(2*N)))
    if interval=='empirical_bernstein':
        # Maurer & Pontil (2009), Theorem 4 is one-sided; a two-sided (1-delta) radius uses delta/2 per tail, i.e. ln(4/delta).
        if N<2: return 1.
        V=float(np.var(values,ddof=1)); ln=np.log(4/delta)
        return float(np.sqrt(2*V*ln/N)+7*ln/(3*(N-1)))
    raise ValueError('Unknown interval')

def confidence_limits(values,delta,interval):
    """Two-sided (1-delta) limits for the mean of i.i.d. values in [0,1]; exact Clopper-Pearson for 0/1 data."""
    values=np.asarray(values,float); N=len(values); e=float(values.mean())
    if interval=='exact_binomial':
        from scipy.stats import beta
        if not np.all((values==0)|(values==1)): raise ValueError('Exact binomial limits need 0/1 data')
        k=int(values.sum()); lo=0. if k==0 else float(beta.ppf(delta/2,k,N-k+1)); hi=1. if k==N else float(beta.ppf(1-delta/2,k+1,N-k))
        return lo,hi
    eps=radius(values,delta,interval); return max(0.,e-eps),min(1.,e+eps)

def solve_bounds(L,obs_lo,obs_hi,aud_feats,aud_lo,aud_hi):
    words,X,Y,eq,beq,target=_lp_setup(L); ub=[]; b=[]
    for obs,lo,hi in zip(itertools.product(range(2),repeat=L),obs_lo,obs_hi):
        f=np.all(Y==obs,axis=1).astype(float); ub.extend([f,-f]); b.extend([hi,-lo])
    for f,lo,hi in zip(aud_feats,aud_lo,aud_hi): ub.extend([f,-f]); b.extend([hi,-lo])
    answers=[]
    for sign in [1,-1]:
        r=linprog(sign*target,A_ub=ub,b_ub=b,A_eq=eq,b_eq=beq,bounds=(0,None),method='highs')
        if not r.success: raise lp_failure(r)
        if np.max(np.asarray(ub)@r.x-b)>1e-7 or np.max(abs(np.asarray(eq)@r.x-beq))>1e-7:
            raise AssertionError('LP residual')
        answers.append(float(target@r.x))
    return answers

def bounds(y,audit_x,audit_y,alpha=.05,cross=None,moments='neighbor',interval='hoeffding',exact_obs=None,exact_audit=None):
    """exact_obs / exact_audit: replace the sampled observation or audit intervals by the population values,
    which decomposes the finite-sample width into observation-sampling and audit-sampling slack."""
    if cross is not None: moments='neighbor' if cross else 'marginal'
    L=y.shape[1]; words,X,Y,eq,beq,target=_lp_setup(L)
    aud=audit_values(audit_x,audit_y,L,moments); feats=moment_features(X,Y,L,moments)
    # One union allocation over every included empirical constraint (exact constraints are not counted).
    M=(0 if exact_obs is not None else 2**L)+(0 if exact_audit is not None else len(aud)); delta=alpha/max(M,1)
    obs_lo=[];obs_hi=[]
    for i,obs in enumerate(itertools.product(range(2),repeat=L)):
        if exact_obs is not None: obs_lo.append(max(0.,exact_obs[i]-1e-9)); obs_hi.append(min(1.,exact_obs[i]+1e-9)); continue
        lo,hi=confidence_limits(np.all(y==obs,axis=1).astype(float),delta,interval); obs_lo.append(lo); obs_hi.append(hi)
    aud_lo=[];aud_hi=[]
    for i,v in enumerate(aud):
        if exact_audit is not None: aud_lo.append(max(0.,exact_audit[i]-1e-9)); aud_hi.append(min(1.,exact_audit[i]+1e-9)); continue
        lo,hi=confidence_limits(v,delta,interval); aud_lo.append(lo); aud_hi.append(hi)
    return solve_bounds(L,obs_lo,obs_hi,feats,aud_lo,aud_hi)

def population_moments(stay,detect,L,moments):
    p=population_law(stay,detect,L); words,X,Y,eq,beq,target=_lp_setup(L)
    obs=[float(np.all(Y==o,axis=1)@p) for o in itertools.product(range(2),repeat=L)]
    return obs,[float(f@p) for f in moment_features(X,Y,L,moments)]

def population_law(stay,detect,L):
    """Exact word distribution over the latent alphabet {0:(X=0),1:(X=1,Y=0),2:(X=1,Y=1)}."""
    words=list(itertools.product(range(3),repeat=L)); index={w:i for i,w in enumerate(words)}; p=np.zeros(len(words))
    for prev in [0,1]:
        for xs in itertools.product([0,1],repeat=L):
            px=.5*(stay if xs[0]==prev else 1-stay)
            for t in range(1,L): px*=stay if xs[t]==xs[t-1] else 1-stay
            preds=[prev]+list(xs[:-1])
            for ds in itertools.product([0,1],repeat=L):
                pr=px; w=[]
                for t in range(L):
                    if xs[t]==0:
                        if ds[t]: pr=0.; break
                        w.append(0)
                    else:
                        d=detect[preds[t]]; pr*=d if ds[t] else 1-d; w.append(2 if ds[t] else 1)
                if pr>0: p[index[tuple(w)]]+=pr
    return p

def population_bounds(stay,detect,L,moments):
    """Identified set with exact moments (no sampling): the limit every interval method approaches."""
    p=population_law(stay,detect,L); words,X,Y,eq,beq,target=_lp_setup(L)
    obs=[float(np.all(Y==o,axis=1)@p) for o in itertools.product(range(2),repeat=L)]
    feats=moment_features(X,Y,L,moments); aud=[float(f@p) for f in feats]
    tiny=1e-9
    return solve_bounds(L,[max(0,v-tiny) for v in obs],[min(1,v+tiny) for v in obs],feats,[max(0,v-tiny) for v in aud],[min(1,v+tiny) for v in aud])

def run(c,out):
    if c.get('experiment'):
        # Configs with an 'experiment' key run the six-variable comparison or the violation study (w3w2.py).
        from . import w3w2
        return w3w2.run(c,out)
    rows=AuditRows(out); detect=[.95,.2]
    moment_sets=c.get('moment_sets',['marginal','neighbor']); intervals=c.get('intervals',['hoeffding']); lengths=c.get('lengths',[3])
    population={}
    for seed in c['seeds']:
        rng=np.random.default_rng(seed)
        for stay in c['stay']:
            for L in lengths:
                _,y=sample_episodes(rng,c['episodes'],L,stay,detect)
                for budget in c['audit_budgets']:
                    x,ay=sample_episodes(rng,budget,L,stay,detect)
                    for moments in moment_sets:
                        key=(stay,L,moments)
                        if key not in population: population[key]=population_bounds(stay,detect,L,moments)
                        plo,phi=population[key]
                        for interval in intervals:
                            sources=['both_sampled']+(list(c.get('decomposition_sources',[])) if moments in c.get('decomposition_moments',[]) else [])
                            for source in sources:
                                exact_obs=exact_aud=None
                                if source!='both_sampled':
                                    pobs,paud=population_moments(stay,detect,L,moments)
                                    if source=='exact_observations': exact_obs=pobs
                                    elif source=='exact_audit': exact_aud=paud
                                    else: raise ValueError('Unknown decomposition source')
                                name=f'{moments}/{interval}'+('' if source=='both_sampled' else f'/{source}')
                                base=dict(seed=seed,stay=stay,L=L,audits=budget,moments=moments,interval=interval,source=source,method=name,
                                          truth=2*stay-1,population_low=plo,population_high=phi,population_width=phi-plo)
                                try: lo,hi=bounds(y,x[:,AUDIT_T],ay,moments=moments,interval=interval,exact_obs=exact_obs,exact_audit=exact_aud)
                                except InfeasibleConfidenceSet as error:
                                    # Empty confidence set: keep the flag and report the conservative [-1,1] fallback.
                                    # A SolverFailure is not caught: the job fails and its manifest records the solver status.
                                    rows.append(dict(**base,status='OK',empty_confidence_set=True,error=str(error),low=-1.,high=1.,width=2.,covered=True,
                                                     concentration_slack=2.-(phi-plo))); continue
                                rows.append(dict(**base,status='OK',empty_confidence_set=False,low=lo,high=hi,width=hi-lo,covered=bool(lo<=2*stay-1<=hi),
                                                 concentration_slack=(hi-lo)-(phi-plo)))
                    np.savez_compressed(out/f'{seed}_{stay}_{L}_{budget}.npz',y=y,audit_x=x[:,AUDIT_T],audit_y=ay)
    return rows
