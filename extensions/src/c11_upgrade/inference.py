"""Finite-sample outer bounds with separate calibration and moment error budgets."""
from itertools import product
import numpy as np
from scipy.stats import beta
from .population import features, law_limits, fp_mass, word_problem, lp_interval, markov_law


def cp_interval(k, n, alpha=.05):
    if not isinstance(k,(int,np.integer)) or not isinstance(n,(int,np.integer)) or n<=0 or not 0<=k<=n or not 0<alpha<1:
        raise ValueError("Require integer 0 <= k <= n, n>0, 0<alpha<1")
    return (0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k)))


def cp_upper(k, n, alpha=.01):
    cp_interval(k,n,alpha)  # Common input validation; one-sided tail below.
    return 1. if k==n else float(beta.ppf(1-alpha,k+1,n-k))


def seed_rng(condition_id, seed, cohort_id):
    return np.random.default_rng(np.random.SeedSequence([20261014,11,2,condition_id,seed,cohort_id]))


def sample_episodes(rng, n, stay=.5, detect=(.95,.2), phi=0., prevalence=None, rho=.6):
    """Independent episodes, with conditional independence used only to generate data."""
    m=.5 if prevalence is None else float(prevalence)
    if not (0<=m<=1 and 0<=stay<=1 and 0<=rho<=1 and 0<=phi<=1
            and len(detect)==2 and all(0<=d<=1 for d in detect)):
        raise ValueError("Invalid generator probability")
    if prevalence is None: p01=p10=1-float(stay)
    else: p01=m*(1-rho); p10=(1-m)*(1-rho)
    prev=(rng.random(n)<m).astype(int)
    x=np.zeros((n,3),int); y=np.zeros((n,3),int)
    for t in range(3):
        prob_x=np.where(prev==0,p01,1-p10)
        x[:,t]=(rng.random(n)<prob_x).astype(int)
        py=np.where(x[:,t]==1,np.asarray(detect)[prev],phi)
        y[:,t]=(rng.random(n)<py).astype(int)
        prev=x[:,t]
    return x,y


def sample_counts(y, x1=None):
    y=np.asarray(y)
    if y.ndim!=2 or y.shape[1]!=3 or len(y)==0 or not np.isin(y,[0,1]).all():
        raise ValueError("Require nonempty binary N x 3 observations")
    ids=y@np.array([4,2,1]); out={f"o{k}":int(np.sum(ids==k)) for k in range(8)}
    if x1 is not None:
        x1=np.asarray(x1)
        if x1.shape!=(len(y),) or not np.isin(x1,[0,1]).all(): raise ValueError("Invalid central gold")
        values={"m":x1,"a":y[:,0]*x1,"b":x1*y[:,2],"j":x1*y[:,1],
                "cL":y[:,0]*x1*y[:,1],"cR":x1*y[:,1]*y[:,2]}
        out.update({k:int(v.sum()) for k,v in values.items()})
        out.update({f"u{k}":int(np.sum(8*x1+ids==k)) for k in range(16)})
    return out


def moment_limits(obs_y, audit_x1, audit_y, alpha=.04):
    oc,ac=sample_counts(obs_y),sample_counts(audit_y,audit_x1)
    return {**{f"o{k}":cp_interval(oc[f"o{k}"],len(obs_y),alpha/11) for k in range(8)},
            **{key:cp_interval(ac[key],len(audit_y),alpha/11) for key in ("m","a","b")}}


def shared_box_comparison(obs_y, audit_x1, audit_y, epsilon):
    """One 24-cell box; identical aggregated moments are reused at every level."""
    oc,ac=sample_counts(obs_y),sample_counts(audit_y,audit_x1)
    bounds={**{f"o{k}":cp_interval(oc[f"o{k}"],len(obs_y),.04/24) for k in range(8)},
            **{f"u{k}":cp_interval(ac[f"u{k}"],len(audit_y),.04/24) for k in range(16)}}
    for name in ("m","a","b","j","cL","cR"):
        keys=[]
        for x,y0,y1,y2 in product(range(2),repeat=4):
            value={"m":x,"a":y0*x,"b":x*y2,"j":x*y1,"cL":y0*x*y1,"cR":x*y1*y2}[name]
            if value: keys.append(f"u{8*x+4*y0+2*y1+y2}")
        bounds[name]=(max(0.,sum(bounds[k][0] for k in keys)),min(1.,sum(bounds[k][1] for k in keys)))
    result={name:lp_interval(word_problem(bounds,epsilon,info=name)) for name in ("N","N6","N8","F")}
    violations=[]; unchecked=[]
    for weak,strong in (("N","N6"),("N6","N8"),("N8","F")):
        a,b=result[weak],result[strong]
        if "solver_failure" in (a["status"],b["status"]): unchecked.append([weak,strong])  # a solver failure leaves the pair unchecked; it is not counted as "no violation"
        if b["status"]=="ok" and a["status"]=="empty": violations.append([weak,strong,"empty outer"])
        if a["status"]==b["status"]=="ok" and (a["lower"]>b["lower"]+1e-8 or a["upper"]<b["upper"]-1e-8):
            violations.append([weak,strong,"endpoint order"])
    return {"intervals":result,"nesting_violations":violations,"nesting_unchecked":unchecked,"limits":bounds}


def condition_grid():
    out=[]
    for family,totals in (("primary",[1000]),("secondary",[200,4000])):
        for total,s,phi in product(totals,[.2,.5,.8],[0.,.02,.10]):
            out.append(dict(family=family,total_gold=total,stay=s,phi=phi,detect=[.95,.2]))
    for m in (.2,.8):
        out.append(dict(family="prevalence",total_gold=1000,stay=.5,phi=.10,detect=[.95,.2],prevalence=m,rho=.6))
    for i,c in enumerate(out): c["condition_id"]=i
    return out


def population_reference(condition):
    args={k:condition[k] for k in ("stay","detect","phi","prevalence","rho") if k in condition}
    p=markov_law(**args); eps=float(fp_mass(p)); limits=law_limits(p)
    target=float(features()[3]@p.ravel())
    identified=lp_interval(word_problem(limits,eps))
    if identified["status"]!="ok": raise RuntimeError("Invalid oracle law")
    return dict(theta=target,f=eps,interval=identified)


def infer_intervals(obs_y, audit_x1, audit_y, calibration_x1, calibration_y1, true_f):
    """Only central gold is accepted. Oracle f is explicitly tagged and isolated."""
    calibration_x1=np.asarray(calibration_x1); calibration_y1=np.asarray(calibration_y1)
    if calibration_x1.shape!=calibration_y1.shape or calibration_x1.ndim!=1 or not np.isin(calibration_x1,[0,1]).all() or not np.isin(calibration_y1,[0,1]).all():
        raise ValueError("Invalid calibration records")
    k=int(np.sum((calibration_x1==0)&(calibration_y1==1))); n=len(calibration_x1)
    upper=cp_upper(k,n,.01)
    results={}
    for name,eps,alpha,states in (("naive_zero",0.,.05,3),("oracle_joint",true_f,.05,4),
                                  ("learned_upper",upper,.04,4),("plugin_joint",k/n,.05,4)):
        limits=moment_limits(obs_y,audit_x1,audit_y,alpha)
        result=lp_interval(word_problem(limits,eps,states=states))
        result.update(epsilon=eps,calibration_k=k,calibration_n=n)
        results[name]=result
    return results


def run_replicate(condition, seed, reference=None, nesting=False):
    cid=condition["condition_id"]; n=condition["total_gold"]//2
    args={k:condition[k] for k in ("stay","detect","phi","prevalence","rho") if k in condition}
    _,oy=sample_episodes(seed_rng(cid,seed,0),10000,**args)
    ax,ay=sample_episodes(seed_rng(cid,seed,1),n,**args)
    cx,cy=sample_episodes(seed_rng(cid,seed,2),n,**args)
    ref=population_reference(condition) if reference is None else reference
    results=infer_intervals(oy,ax[:,1],ay,cx[:,1],cy[:,1],ref["f"])
    oracle_width=(results["oracle_joint"]["upper"]-results["oracle_joint"]["lower"]
                  if results["oracle_joint"]["status"]=="ok" else None)
    metrics=[]
    for name,r in results.items():
        valid=r["status"]=="ok"; empty=r["status"]=="empty"
        lo,hi=(r["lower"],r["upper"]) if valid else (-1.,1.)
        contained=valid and lo<=ref["interval"]["lower"]+1e-8 and hi>=ref["interval"]["upper"]-1e-8
        truth=valid and lo-1e-8<=ref["theta"]<=hi+1e-8
        # A solver failure also reports the fallback [-1,1] (width 2), so fallback-inclusive widths cannot shrink by
        # dropping failures; its coverage counts as a miss.
        failed=r["status"]=="solver_failure"
        width=hi-lo if valid or empty or failed else None
        metrics.append(dict(condition_id=cid,family=condition["family"],seed=seed,method=name,
            **r,theta=ref["theta"],true_f=ref["f"],population_lower=ref["interval"]["lower"],
            population_upper=ref["interval"]["upper"],contains_population=bool(contained),
            covers_truth=bool(truth),fallback_contains_population=bool(contained or empty),
            fallback_covers_truth=bool(truth or empty),width=width,empty_feasible=empty,
            solver_failure=failed,fallback_used=empty or failed,
            epsilon_covers_f=r["epsilon"]>=ref["f"]-1e-12,upper_minus_f=r["epsilon"]-ref["f"],
            width_minus_oracle=None if width is None or oracle_width is None else width-oracle_width))
    shared=shared_box_comparison(oy,ax[:,1],ay,results["learned_upper"]["epsilon"]) if nesting else None
    return {"condition":condition,"seed":seed,"metrics":metrics,"shared_box":shared}


def coverage_decision(intervals):
    """Refutation takes precedence over the weaker loss-exclusion threshold."""
    if any(hi<.95 for lo,hi in intervals): return "refutation_signal"
    if len(intervals)==9 and all(lo>=.93 for lo,hi in intervals): return "large_loss_excluded"
    return "inconclusive"


def summarize(rows, bootstrap_reps=10000, smoke=False):
    groups={}
    for r in rows: groups.setdefault((r["condition_id"],r["method"]),[]).append(r)
    output=[]; primary_ci=[]
    for (cid,method),group in sorted(groups.items()):
        n=len(group); hits=sum(r["contains_population"] for r in group)
        primary=group[0]["family"]=="primary" and method=="learned_upper"
        ci=cp_interval(hits,n,.05/9 if primary else .05)
        if primary: primary_ci.append(ci)
        valid=[r for r in group if r["status"]=="ok"]
        diffs=np.array([r["width_minus_oracle"] for r in group if r["width_minus_oracle"] is not None])
        boot_ci=None
        if len(diffs):
            rng=seed_rng(cid,20261014,9)
            # Batch to bound memory even for the full planned grid.
            means=[]
            for start in range(0,bootstrap_reps,100):
                samples=rng.choice(diffs,size=(min(100,bootstrap_reps-start),len(diffs)),replace=True)
                means.extend(samples.mean(axis=1))
            boot_ci=np.quantile(means,[.025,.975]).tolist()
        output.append(dict(condition_id=cid,method=method,family=group[0]["family"],n=n,
            contains_population=hits/n,mc_ci=list(ci),covers_truth=sum(r["covers_truth"] for r in group)/n,
            truth_mc_ci=list(cp_interval(sum(r["covers_truth"] for r in group),n)),
            fallback_contains_population=sum(r["fallback_contains_population"] for r in group)/n,
            empty_rate=sum(r["empty_feasible"] for r in group)/n,
            solver_failure_rate=sum(r["solver_failure"] for r in group)/n,
            fallback_rate=sum(r["fallback_used"] for r in group)/n,
            nonempty_coverage=None if not valid else sum(r["contains_population"] for r in valid)/len(valid),
            mean_width=None if not any(r["width"] is not None for r in group) else float(np.mean([r["width"] for r in group if r["width"] is not None])),
            epsilon_coverage=sum(r["epsilon_covers_f"] for r in group)/n,
            mean_upper_minus_f=float(np.mean([r["upper_minus_f"] for r in group])),
            mean_width_minus_oracle=None if not len(diffs) else float(diffs.mean()),
            paired_bootstrap_95=boot_ci,paired_complete_n=len(diffs)))
    precision_cells=[r for r in output if r["method"]=="learned_upper" and r["family"]=="primary"
                     and condition_grid()[r["condition_id"]]["phi"]>0]
    practical=sum(r["mean_width"] is not None and r["mean_width"]<1.8 and r["fallback_rate"]<.05 for r in precision_cells)>=3
    expected=set(range(410001,411001))
    primary_complete=all((cid,"learned_upper") in groups
                         and len(groups[cid,"learned_upper"])==1000
                         and {r["seed"] for r in groups[cid,"learned_upper"]}==expected for cid in range(9))
    prefix="smoke_not_confirmatory" if smoke else (None if primary_complete else "incomplete_protocol")
    return {"smoke_only":smoke,"primary_grid_complete":primary_complete,
            "coverage_decision":prefix or coverage_decision(primary_ci),
            "practical_precision":prefix or ("nonvacuous_tested_cells" if practical else "criterion_not_met"),"groups":output}
