"""Overlapping-window experiment: intervals from windows cut from one record, compared with independent episodes, thinning and blocking. Nothing runs on import.

Block Hoeffding intervals are heuristic under serial dependence, not a theorem
of independent blocks. Audit empty blocks are excluded, never filled with zero.
"""
import math
import numpy as np
from .inference import cp_interval, cp_upper, sample_episodes
from .population import markov_law, lp_interval
from .addendum_core import rng, information_limits, information_names, make_problem, population_references, interval_metrics

METHODS = ("iid", "naive_overlap", "thin_3", "thin_10", "block_50", "block_200")


def condition_grid():
    result = [dict(condition_id=i, family="primary", stay=s, phi=phi, detect=[.95,.2])
              for i, (s, phi) in enumerate((s,p) for s in (.2,.5,.8) for p in (0.,.02))]
    result.append(dict(condition_id=6, family="auxiliary", prevalence=.2, rho=.6,
                       stay=.5, phi=.02, detect=[.95,.2]))
    return result


def _binary(a, width=None):
    a = np.asarray(a)
    if (a.ndim != (1 if width is None else 2) or len(a) == 0
            or (width is not None and a.shape[1] != width) or not np.isin(a,[0,1]).all()):
        raise ValueError("Require nonempty binary observations")
    return a.astype(int, copy=False)


def _indices(indices, frame_size, count=None):
    z = np.asarray(indices)
    if (z.ndim != 1 or not np.issubdtype(z.dtype,np.integer) or len(z)==0
            or len(np.unique(z)) != len(z) or np.any(z<0) or np.any(z>=frame_size)
            or (count is not None and len(z) != count)):
        raise ValueError("Invalid unique frame indices")
    return z.astype(int,copy=False)


def _blocks(block_ids):
    a=np.asarray(block_ids)
    if a.ndim!=1 or not len(a): raise ValueError("Require nonempty 1D block ids")
    if a.dtype.kind in 'fc' and not np.isfinite(a).all(): raise ValueError("Nonfinite block id")
    if any(v is None for v in a.tolist()): raise ValueError("Missing block id")
    labels, inverse, counts=np.unique(a,return_inverse=True,return_counts=True)
    return labels,inverse,counts


def priority_select(priorities, eligible, budget):
    """Filter one continuous-priority vector, break exact ties by global index."""
    p=np.asarray(priorities,float)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Invalid priority vector")
    e=_indices(eligible,len(p))
    if not isinstance(budget,(int,np.integer)) or not 0<budget<=len(e):
        raise ValueError("Insufficient eligible centers for fixed audit budget")
    return e[np.lexsort((e,p[e]))[:budget]]


def consecutive_windows(x,y):
    x,y=_binary(x),_binary(y)
    if x.shape!=y.shape or len(x)<3: raise ValueError("Matched length >=3 records required")
    return (np.lib.stride_tricks.sliding_window_view(x,3),
            np.lib.stride_tricks.sliding_window_view(y,3))


def generate_record(random, length=10002, stay=.5, detect=(.95,.2), phi=0., prevalence=None, rho=.6):
    """Simulate one stationary latent sequence X. P(Y_t=1) is phi when X_t=0 and detect[X_{t-1}] when X_t=1."""
    if not isinstance(length,(int,np.integer)) or length<3: raise ValueError("Record too short")
    m=.5 if prevalence is None else float(prevalence)
    if not (0<=m<=1 and 0<=stay<=1 and 0<=rho<=1 and 0<=phi<=1
            and len(detect)==2 and all(0<=v<=1 for v in detect)):
        raise ValueError("Invalid generation probability")
    p01=p10=1-float(stay)
    if prevalence is not None: p01=m*(1-rho);p10=(1-m)*(1-rho)
    previous=int(random.random()<m)
    x=np.empty(length,dtype=int);y=np.empty(length,dtype=int)
    for t in range(length):
        x[t]=int(random.random()<(p01 if previous==0 else 1-p10))
        y[t]=int(random.random()<(detect[previous] if x[t] else phi))
        previous=x[t]
    return x,y


def _features(y,x=None):
    ids=y@np.array([4,2,1])
    out={f"o{k}":(ids==k).astype(float) for k in range(8)}
    if x is not None:
        out.update(m=x.astype(float),a=y[:,0]*x,b=x*y[:,2])
        out.update({f"u{k}":(8*x+ids==k).astype(float) for k in range(16)})
    return out


def _block_means(values,inverse,nblocks):
    counts=np.bincount(inverse,minlength=nblocks)
    sums=np.bincount(inverse,weights=values,minlength=nblocks)
    keep=counts>0
    return sums[keep]/counts[keep],counts


def block_moment_limits(obs_y,audit_x1,audit_y,audit_indices,block_ids,alpha=.04,info="N"):
    oy,ay,ax=_binary(obs_y,3),_binary(audit_y,3),_binary(audit_x1)
    if len(ay)!=len(ax): raise ValueError("Mismatched audit gold")
    labels,inv,sizes=_blocks(block_ids)
    if len(inv)!=len(oy): raise ValueError("Block ids must index entire observed frame")
    idx=_indices(audit_indices,len(oy),len(ay))
    if not np.array_equal(ay,oy[idx]): raise ValueError("Audit outcomes differ from indexed frame")
    names=information_names(info)
    if not 0<alpha<1: raise ValueError("Invalid moment error budget")
    delta=alpha/len(names)
    obs=_features(oy);aud=_features(ay,ax)
    limits={};centers={};radii={};audit_counts=None
    for name in names:
        source,mask=(obs[name],inv) if name.startswith("o") else (aud[name],inv[idx])
        means,counts=_block_means(source,mask,len(labels))
        if not len(means): raise ValueError("No nonempty blocks")
        radius=math.sqrt(math.log(2/delta)/(2*len(means)))
        center=float(means.mean())
        limits[name]=(max(0.,center-radius),min(1.,center+radius))
        centers[name]=center;radii[name]=radius
        if not name.startswith("o"): audit_counts=counts
    return limits,dict(block_ids=labels.tolist(),total_blocks=len(labels),block_sizes=sizes.tolist(),
        audit_block_counts=audit_counts.tolist(),nonempty_audit_blocks=int(np.count_nonzero(audit_counts)),
        audit_n=len(ay),observation_n=len(oy),centers=centers,radii=radii,delta=delta,
        weighting="equal_nonempty_block_means",independence_guaranteed=False)


def block_upper(cal_x1,cal_y1,cal_indices,cal_block_ids,error=.01):
    x,y=_binary(cal_x1),_binary(cal_y1)
    if x.shape!=y.shape or not 0<error<1: raise ValueError("Invalid calibration input")
    labels,inv,sizes=_blocks(cal_block_ids)
    idx=_indices(cal_indices,len(inv),len(x))
    values=((x==0)&(y==1)).astype(float)
    means,counts=_block_means(values,inv[idx],len(labels))
    radius=math.sqrt(math.log(1/error)/(2*len(means)))
    center=float(means.mean());upper=min(1.,center+radius)
    return upper,dict(block_ids=labels.tolist(),total_blocks=len(labels),block_sizes=sizes.tolist(),
        calibration_block_counts=counts.tolist(),nonempty_calibration_blocks=len(means),
        calibration_n=len(x),calibration_k=int(values.sum()),center=center,radius=radius,
        weighting="equal_nonempty_block_means",independence_guaranteed=False)


def run_replicate(condition,seed,reference=None):
    """Run one replicate of all six methods for one condition and seed."""
    if seed not in range(440001,441001) and seed not in range(990001,990004):
        raise ValueError("Seed outside the registered experiment/test ranges")
    cid=condition["condition_id"]
    if condition!=condition_grid()[cid]: raise ValueError("Condition differs from condition_grid()")
    args={k:condition[k] for k in ("stay","detect","phi","prevalence","rho") if k in condition}
    ref=population_references(markov_law(**args)) if reference is None else reference
    x,y=generate_record(rng(3,cid,seed,0),**args)
    xw,yw=consecutive_windows(x,y)
    priority=rng(3,cid,seed,1).random(len(yw))
    if condition['phi']>0:
        cx,cy=generate_record(rng(3,cid,seed,2),**args)
        cxw,cyw=consecutive_windows(cx,cy)
        cp=rng(3,cid,seed,3).random(len(cyw))
    metrics=[]
    for method in METHODS:
        diagnostics={};caldiag={}
        if method=='iid':
            _,oy=sample_episodes(rng(3,cid,seed,4),10000,**args)
            ax,ay=sample_episodes(rng(3,cid,seed,5),500,**args);ax1=ax[:,1]
            if condition['phi']>0:
                calx,caly=sample_episodes(rng(3,cid,seed,6),500,**args)
                calx1,caly1=calx[:,1],caly[:,1]
        else:
            k=int(method.split('_')[1]) if method.startswith('thin_') else 1
            eligible=np.arange(0,len(yw),k)
            ai=priority_select(priority,eligible,500)
            oy=yw[eligible];ay=yw[ai];ax1=xw[ai,1]
            diagnostics.update(audit_indices=ai.tolist(),eligible_n=len(eligible),thin_k=k)
            if condition['phi']>0:
                ci=priority_select(cp,np.arange(0,len(cyw),k),500)
                calx1,caly1=cxw[ci,1],cyw[ci,1]
                caldiag.update(calibration_indices=ci.tolist())
        alpha=.05 if condition['phi']==0 else .04
        if method.startswith('block_'):
            ell=int(method.split('_')[1]);blocks=np.arange(len(yw))//ell
            limits,blockdiag=block_moment_limits(yw,ax1,ay,ai,blocks,alpha,'N')
            diagnostics.update(blockdiag)
        else:
            limits=information_limits(oy,ax1,ay,'N',alpha)
        if condition['phi']==0:
            epsilon=0.;caldiag.update(calibration_n=0,calibration_k=None,known_no_fp=True)
        elif method.startswith('block_'):
            epsilon,bd=block_upper(calx1,caly1,ci,np.arange(len(cyw))//ell)
            caldiag.update(bd)
        else:
            nk=int(np.sum((calx1==0)&(caly1==1)))
            epsilon=cp_upper(nk,len(calx1),.01)
            caldiag.update(calibration_n=len(calx1),calibration_k=nk)
        result=lp_interval(make_problem(limits,epsilon,'N'))
        row=dict(condition_id=cid,family=condition['family'],seed=int(seed),method=method,
                 epsilon=epsilon,true_f=ref['f'],theta=ref['theta'],audit_n=len(ay),observation_n=len(oy),
                 diagnostics=diagnostics,calibration=caldiag,limits=limits,**result)
        row.update(interval_metrics(result,ref['intervals']['N'],ref['theta']))
        metrics.append(row)
    return dict(condition=condition,seed=int(seed),metrics=metrics)


def _decision(intervals):
    if any(hi<.95 for lo,hi in intervals): return 'refutation_signal'
    if len(intervals)==6 and all(lo>=.93 for lo,hi in intervals): return 'large_loss_excluded'
    return 'inconclusive'


def summarize(rows,smoke=False,bootstrap_reps=10000):
    """Summarize coverage and width. A method is selected for the recorded-data application only from a complete, duplicate-free grid."""
    if not isinstance(bootstrap_reps,(int,np.integer)) or bootstrap_reps<1: raise ValueError('Invalid bootstrap count')
    groups={}
    for r in rows: groups.setdefault((r['condition_id'],r['method']),[]).append(r)
    expected=set(range(440001,441001))
    def complete(cid,method):
        g=groups.get((cid,method),[])
        return len(g)==1000 and {r['seed'] for r in g}==expected and all(r['contains_population'] is not None for r in g)
    valid_keys={(cid,m) for cid in range(7) for m in METHODS}
    primary_complete=all(complete(cid,m) for cid in range(6) for m in METHODS)
    complete_grid=set(groups)==valid_keys and all(complete(cid,m) for cid,m in valid_keys)
    output=[];ci_map={};width_map={}
    for (cid,method),group in sorted(groups.items()):
        valid=[r for r in group if r['contains_population'] is not None]
        hits=sum(bool(r['contains_population']) for r in valid)
        error=.05/6 if cid<6 and method=='naive_overlap' else (.05/12 if cid<6 and method in ('thin_10','block_200') else .05)
        ci=cp_interval(hits,len(valid),error) if valid else None
        ci_map[cid,method]=ci
        widths=[r['width'] for r in group]
        mean=float(np.mean(widths));width_map[cid,method]=mean
        iidgroup=groups.get((cid,'iid'),[])
        iid={r['seed']:r for r in iidgroup}
        duplicate=len(iid)!=len(iidgroup) or len({r['seed'] for r in group})!=len(group)
        paired=[] if duplicate else [r['width']-iid[r['seed']]['width'] for r in group if r['seed'] in iid]
        boot_ci=None
        if paired:
            generator=rng(3,cid,20261014,9);means=[]
            for start in range(0,bootstrap_reps,100):
                n=min(100,bootstrap_reps-start)
                means.extend(generator.choice(paired,size=(n,len(paired)),replace=True).mean(axis=1))
            boot_ci=np.quantile(means,[.025,.975]).tolist()
        output.append(dict(condition_id=cid,method=method,n=len(group),valid_reference_n=len(valid),
            undefined_reference_n=len(group)-len(valid),contains_population=None if not valid else hits/len(valid),
            contains_population_all_denominator=hits/len(group),mc_ci=None if ci is None else list(ci),mc_error=error,
            mean_width=mean,empty_n=sum(r['status']=='empty' for r in group),
            solver_failure_n=sum(r['status']=='solver_failure' for r in group),
            fallback_n=sum(r.get('fallback_used',r['status']!='ok') for r in group),
            covers_truth=sum(bool(r['covers_truth']) for r in group)/len(group),
            paired_iid_n=len(paired),paired_width_difference=None if not paired else float(np.mean(paired)),
            paired_bootstrap_95=boot_ci,paired_duplicates=duplicate))
    blocked='NOT_EVALUATED_test_or_smoke' if smoke else (None if complete_grid else 'incomplete_protocol')
    h8a=blocked or ('refutation_signal' if any(ci_map[cid,'naive_overlap'][1]<.95 for cid in range(6)) else 'not_detected')
    h8b={method:blocked or _decision([ci_map[cid,method] for cid in range(6)]) for method in ('thin_10','block_200')}
    eligible=[m for m,d in h8b.items() if d=='large_loss_excluded']
    averages={m:float(np.mean([width_map[cid,m] for cid in range(6)])) for m in eligible}
    selected='application_only'
    if len(eligible)==1: selected=eligible[0]
    elif len(eligible)==2:
        selected='thin_10' if averages['thin_10']<=averages['block_200']+1e-8 else 'block_200'
    return dict(primary_complete=primary_complete,complete_grid=complete_grid,smoke_only=bool(smoke),
                h8a=h8a,h8b=h8b,selection=dict(method=selected,eligible=eligible,mean_widths=averages,
                reason=blocked or ('eligible_minimum_width' if eligible else 'no_method_passed'),
                coverage_certification=False),groups=output)
