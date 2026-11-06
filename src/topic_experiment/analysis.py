"""Stratified paired analysis. Never pool unrelated conditions or count correlated rows as seeds."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from .common import write_json

SPEC={
 'C11':('width','seed',['stay','L','audits'],'method')
}

def paired(values_a,values_b):
    units=sorted(set(values_a)&set(values_b)); delta=np.array([np.mean(values_a[k])-np.mean(values_b[k]) for k in units])
    if not len(delta): return dict(units=0,mean_difference=None,interval=None)
    rng=np.random.default_rng(617); interval=None
    if len(delta)>=5: interval=np.quantile(delta[rng.integers(len(delta),size=(4000,len(delta)))].mean(1),[.025,.975]).tolist()
    return dict(units=len(delta),mean_difference=float(delta.mean()),interval=interval,unit_differences=delta.tolist())

def descriptive(topic,rows):
    if topic not in SPEC:
        return dict(topic=topic,kind='exact_oracle_diagnostic',rows=rows,warning='Exact population comparisons; no sampling confidence interval.')
    metric,unit,conditions,method=SPEC[topic]; strata={}
    for r in rows:
        key=tuple(r.get(k,'v1_unspecified') for k in conditions); name=str(r[method])
        g=strata.setdefault(key,{})
        uid=str(r[unit])
        if r[metric] is not None: g.setdefault(name,{}).setdefault(uid,[]).append(r[metric])
    output=[]
    for key,methods in strata.items():
        comparisons=[]
        for a,b in itertools.combinations(sorted(methods),2): comparisons.append(dict(a=a,b=b,**paired(methods[a],methods[b])))
        output.append(dict(condition=dict(zip(conditions,key)),unit_means={name:{u:float(np.mean(v)) for u,v in vals.items()} for name,vals in methods.items()},comparisons=comparisons))
    result=dict(topic=topic,metric=metric,unit=unit,strata=output,
                warning='Paired descriptive bootstrap intervals. No multiplicity-adjusted confirmatory claim; fewer than five units yields no interval.')
    if topic=='C11':
        result['coverage_by_condition']=[]
        for s,L,a,m in sorted({(r['stay'],r.get('L',3),r['audits'],r['method']) for r in rows}):
            vals=[r['covered'] for r in rows if (r['stay'],r.get('L',3),r['audits'],r['method'])==(s,L,a,m)]
            n=len(vals); rate=np.mean(vals); z=1.95996398454; center=(rate+z*z/(2*n))/(1+z*z/n)
            radius=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
            empty=sum(bool(r.get('empty_confidence_set',False)) for r in rows if (r['stay'],r.get('L',3),r['audits'],r['method'])==(s,L,a,m))
            result['coverage_by_condition'].append(dict(stay=s,L=L,audits=a,method=m,covered=sum(vals),total=n,empty_confidence_sets=empty,
                covered_counting_empty_as_miss=sum(vals)-empty,wilson_95=[float(center-radius),float(center+radius)]))
    return result

def analyze(topic,rows):
    from collections import Counter,defaultdict
    failures=[r for r in rows if r.get('status','OK')!='OK']
    if topic=='C11' and rows and rows[0].get('experiment'):
        # Jobs with an 'experiment' key hold one seed; the cross-seed analysis is in manuscript/supplement/sixvar_violations/.
        from .w3w2 import job_summary
        result=job_summary(rows)
        result['execution_quality']=dict(total_rows=len(rows),valid_rows=len(rows)-len(failures),non_success_rows=len(failures),statuses=dict(Counter(r.get('status','OK') for r in rows)),failures=failures,claim_eligible=not failures)
        result['claim_status']='PER_JOB_SUMMARY_NOT_A_CLAIM'
        return result
    valid=[r for r in rows if r.get('status','OK')=='OK']
    result=descriptive(topic,valid) if valid else dict(topic=topic,strata=[])
    result['execution_quality']=dict(total_rows=len(rows),valid_rows=len(valid),non_success_rows=len(failures),statuses=dict(Counter(r.get('status','OK') for r in rows)),failures=failures,claim_eligible=not failures)
    primary=[]
    if topic=='C11':
        groups=defaultdict(dict)
        for r in valid:groups[(r['seed'],r['stay'],r.get('L',3),r['audits'])][r['method']]=r
        def contrast(name,a,b,budget=500,L=3):
            for key,g in groups.items():
                if key[3]==budget and key[2]==L and a in g and b in g:
                    w=g[a]['width'];primary.append(dict(contrast=name,seed=key[0],stay=key[1],L=L,audits=budget,reference=a,candidate=b,relative_width_reduction=(w-g[b]['width'])/max(w,1e-12),
                        population_width_reference=g[a].get('population_width'),population_width_candidate=g[b].get('population_width'),slack_fraction_candidate=(g[b].get('concentration_slack') or 0.)/max(g[b]['width'],1e-12)))
        contrast('neighbor_vs_marginal_hoeffding','marginal/hoeffding','neighbor/hoeffding')
        contrast('bernstein_vs_hoeffding_neighbor','neighbor/hoeffding','neighbor/empirical_bernstein')
        contrast('full_vs_neighbor_bernstein','neighbor/empirical_bernstein','full/empirical_bernstein')
        contrast('full_vs_neighbor_bernstein_L4','neighbor/empirical_bernstein','full/empirical_bernstein',L=4)
        contrast('exact_binomial_vs_bernstein_neighbor','neighbor/empirical_bernstein','neighbor/exact_binomial')
        contrast('exact_binomial_vs_hoeffding_neighbor','neighbor/hoeffding','neighbor/exact_binomial')
        # Slack decomposition: width(both sampled) - width(population) = observation slack + audit slack (+ interaction).
        decomposition=[]
        for key,g in groups.items():
            for interval in ['hoeffding','empirical_bernstein','exact_binomial']:
                b=g.get(f'neighbor/{interval}');eo=g.get(f'neighbor/{interval}/exact_observations');ea=g.get(f'neighbor/{interval}/exact_audit')
                if not (b and eo and ea):continue
                pop=b.get('population_width') or 0.;total=b['width']-pop
                decomposition.append(dict(seed=key[0],stay=key[1],L=key[2],audits=key[3],interval=interval,total_slack=total,
                    audit_slack=eo['width']-pop,observation_slack=ea['width']-pop,audit_slack_fraction=None if total<=1e-12 else (eo['width']-pop)/total))
        result['slack_decomposition']=decomposition
        result['primary_estimand']='At audit budget 500 and L=3: empirical-Bernstein vs Hoeffding width for the neighbor audit (threshold .25 in >=2 stay strata, coverage >=.95 everywhere); full vs neighbor audit under empirical Bernstein (threshold .10 in >=2 stay strata); neighbor vs marginal audit under Hoeffding; exact Clopper-Pearson vs empirical-Bernstein and Hoeffding intervals; widths decomposed into the population identified width plus concentration slack, and the slack into observation-sampling and audit-sampling parts'
    result['primary_analysis']=primary
    result['claim_status']='INCONCLUSIVE_WITH_FAILURES' if failures else 'DESCRIPTIVE_ONLY'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--topic',required=True); p.add_argument('--rows',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    if Path(a.out).exists():p.error('Refusing to overwrite analysis')
    write_json(a.out,analyze(a.topic,json.loads(Path(a.rows).read_text())))
