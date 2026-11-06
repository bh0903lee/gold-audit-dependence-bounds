"""Command-line runner for the identification checks (d1), the four-state finite-sample simulation (d2) and the data-contract check (gate). Each run writes to a new, empty directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from fractions import Fraction as Q
from itertools import product
import numpy as np
import scipy
from .population import (features, periodic_law, markov_law, law_limits, fp_mass,
                         word_problem, lp_interval, full_h, pair_from_moments,
                         glue_triple, check_gluing, exact_certificate, closed_form)
from .inference import condition_grid, population_reference, run_replicate, summarize
from .adapter import validate_dataset

ROOT=Path(__file__).resolve().parents[2]


def validate_config():
    config=json.loads((ROOT/"configs/protocol.json").read_text())
    expected={"protocol_version":"0.2","alphabet_xy":[[0,0],[1,0],[1,1],[0,1]],
        "length_primary":3,"length_secondary":4,"alpha_total":.05,"alpha_fp":.01,"alpha_moments":.04,
        "n_obs":10000,"d1_recovery_seeds":[300001,300128],"d1_rational_seeds":[310001,310512],
        "d1_l4_rational_seeds":[310001,310064],"d1_smoke_rational_seeds":[990001,990003],
        "d2_primary_seeds":[410001,411000],"d2_secondary_seeds":[420001,420200],
        "d2_prevalence_seeds":[430001,430200],"d2_smoke_seeds":[990001,990003],
        "d2_smoke_condition_ids":[0,4,8],"d2_full_replicates":13000,"bootstrap_repetitions":10000,
        "float_equivalence_tolerance":1e-8,"float_witness_screen":1e-6,"workers":1,"threads_per_worker":1}
    for key,value in expected.items():
        if config.get(key)!=value: raise ValueError(f"Config/implementation mismatch: {key}")
    return config


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def freeze(out, mode):
    """Write code/config hashes before the runner creates scientific output."""
    validate_config()
    out=Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError("Output directory must be empty; never overwrite a run")
    out.mkdir(parents=True,exist_ok=True)
    files=list((ROOT/"src").rglob("*.py"))+list((ROOT/"configs").glob("*.json"))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    write_json(out/"manifest.json",dict(mode=mode,created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        hashes=hashes,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
        threads={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
        run_complete=False))
    return out


def quota_exceeded(start, out=None):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    config=validate_config()
    used=sum(p.stat().st_size for p in Path(out).rglob("*") if p.is_file()) if out else 0
    return (time.monotonic()-start>config["max_wall_seconds"] or usage.ru_utime+usage.ru_stime>config["max_cpu_seconds"]
            or usage.ru_maxrss>config["max_memory_gb"]*1024**2 or used>config["max_output_gb"]*1024**3)


def completion(out,summary):
    write_json(out/"summary.json",summary)
    manifest=json.loads((out/"manifest.json").read_text())
    manifest["run_complete"]=summary.get("complete",False)
    manifest["output_hashes"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.name!="manifest.json"}
    write_json(out/"manifest.json",manifest)


def d1_bank(full=False):
    for i,(s,det) in enumerate(product([.2,.5,.8],[(.95,.2),(.8,.8)])):
        for length in (3,4): yield f"recovery_markov_{i}",None,length,markov_law(s,det,0.,length)
    seeds=range(300001,300129) if full else range(990001,990004)
    for seed in seeds: yield "recovery_cyclic",seed,3,periodic_law(seed,alphabet=3)
    settings=list(product([.2,.5,.8],[(.95,.2),(.8,.8)],[.002,.01,.05,.10]))
    for i,(s,det,phi) in enumerate(settings if full else settings[:1]):
        yield f"two_sided_{i}",None,3,markov_law(s,det,phi)
    seeds=range(310001,310513) if full else range(990001,990004)
    for seed in seeds:
        yield "rational",seed,3,periodic_law(seed)
        if seed<310065 and full: yield "rational_L4",seed,4,periodic_law(seed,length=4)


def d1(out, full=False):
    start=time.monotonic(); out=freeze(out,"d1_full" if full else "d1_smoke")
    counters={"law_windows":0,"comparisons":0,"lp_failures":0,"equivalence_failures":0,
              "recovery_failures":0,"nesting_failures":0,"exact_certificate_failures":0}
    selected=None; gluing_saved=False; complete=True
    with (out/"population.jsonl").open("w") as stream:
        for family,seed,length,p in d1_bank(full):
            if quota_exceeded(start,out): complete=False; break
            counters["law_windows"]+=1
            limits=law_limits(p,length); f=fp_mass(p,length)
            epsilons=sorted(set([f,min(Q(1),f+Q(1,200)),min(Q(1),f+Q(1,50)),min(Q(1),f+Q(1,10))]))
            previous={}
            h=full_h(p) if length==3 else None
            pair=pair_from_moments(limits) if length==3 else None
            pair_result=lp_interval(pair,return_solutions=True) if pair else None
            if pair_result is not None and pair_result["status"]!="ok": counters["pair_failures"]=counters.get("pair_failures",0)+1  # count failures instead of dropping them
            for eps in epsilons:
                results={}; problems={}
                for info in ("N","N6","N7L","N7R","N8","F"):
                    problem=word_problem(limits,eps,info,length); r=lp_interval(problem,return_solutions=True)
                    problems[info]=problem; results[info]=r
                    if r["status"]!="ok": counters["lp_failures"]+=1
                    if info in previous and previous[info]["status"]==r["status"]=="ok":
                        if r["lower"]>previous[info]["lower"]+1e-8 or r["upper"]<previous[info]["upper"]-1e-8:
                            counters["nesting_failures"]+=1
                previous=results
                equivalence_error=None; recovery_error=None
                if length==3 and not (all(results[k]["status"]=="ok" for k in ("N8","F")) and pair_result["status"]=="ok"):
                    counters["equivalence_unchecked"]=counters.get("equivalence_unchecked",0)+1
                    if f==0 and eps==0: counters["recovery_unchecked"]=counters.get("recovery_unchecked",0)+1
                if length==3 and all(results[k]["status"]=="ok" for k in ("N8","F")) and pair_result["status"]=="ok":
                    n8_error=max(abs(results["N8"][side]-results["F"][side]) for side in ("lower","upper"))
                    m8_error=max(abs(pair_result[side]-results["F"][side]) for side in ("lower","upper"))
                    equivalence_error=max(n8_error,m8_error)
                    counters["equivalence_failures"]+=int(equivalence_error>1e-8)
                    counters["n8_word_equivalence_failures"]=counters.get("n8_word_equivalence_failures",0)+int(n8_error>1e-8)
                    counters["m8_pair_equivalence_failures"]=counters.get("m8_pair_equivalence_failures",0)+int(m8_error>1e-8)
                    if f==0 and eps==0:
                        closed=closed_form(limits)
                        recovery_error=max(abs(results[k][side]-closed[j]) for k in results for j,side in enumerate(("lower","upper")))
                        counters["recovery_failures"]+=int(recovery_error>1e-8)
                gap=None
                if results["F"]["status"]==results["N6"]["status"]=="ok":
                    gap=dict(lower=results["F"]["lower"]-results["N6"]["lower"],upper=results["N6"]["upper"]-results["F"]["upper"])
                # Only the predefined cyclic seed order can select the witness.
                if full and selected is None and family=="rational" and gap and max(gap.values())>1e-6:
                    certificates={}
                    for name in ("F","N6"):
                        certificates[name]=[exact_certificate(problems[name],sol,sign) for sol,sign in zip(results[name]["solutions"],(1,-1))]
                    verified=all(c["verified"] for cs in certificates.values() for c in cs)
                    exact_gaps=None
                    if verified:
                        exact_gaps={"lower":str(Q(certificates["F"][0]["objective"])-Q(certificates["N6"][0]["objective"])),
                                    "upper":str(Q(certificates["N6"][1]["objective"])-Q(certificates["F"][1]["objective"]))}
                        verified=max(map(Q,exact_gaps.values()))>0
                    selected=dict(seed=seed,epsilon=str(eps),verified=verified,gaps=exact_gaps,certificates=certificates,
                                  law=[str(v) for v in p.ravel()],selection="first eligible seed, then smallest epsilon")
                    write_json(out/"gap_certificate.json",selected)
                    counters["exact_certificate_failures"]+=int(not verified)
                if not gluing_saved and family=="rational" and f>0 and pair_result["status"]=="ok":
                    certificates=[]
                    for sol,sign in zip(pair_result["solutions"],(1,-1)):
                        cert=exact_certificate(pair,sol,sign)
                        if cert["verified"]:
                            k=np.array([Q(v) for v in cert["primal"]],dtype=object).reshape(4,4)
                            t=glue_triple(k,h); checks=check_gluing(k,h,t,exact=True)
                            cert.update(gluing_checks=checks,full_h=[str(v) for v in h.ravel()],triple=[str(v) for v in t.ravel()])
                            cert["verified"]=cert["verified"] and all(checks.values())
                        certificates.append(cert)
                    write_json(out/"gluing_certificate.json",dict(seed=seed,certificates=certificates))
                    gluing_saved=True
                    counters["exact_certificate_failures"]+=sum(not c["verified"] for c in certificates)
                clean={k:{a:b for a,b in v.items() if a!="solutions"} for k,v in results.items()}
                record=dict(family=family,seed=seed,length=length,f=str(f),epsilon=str(eps),intervals=clean,
                            pair=None if pair_result is None else {k:v for k,v in pair_result.items() if k!="solutions"},
                            gap_N6_F=gap,equivalence_error=equivalence_error,recovery_error=recovery_error)
                stream.write(json.dumps(record,allow_nan=False)+"\n"); stream.flush(); counters["comparisons"]+=1
    complete=complete and not quota_exceeded(start,out)
    summary=dict(complete=complete,smoke_only=not full,counts=counters,elapsed_seconds=time.monotonic()-start,
                 h1="NOT_EVALUATED_fixture_only" if not full else ("exact_witness" if selected and selected["verified"] else "unverified_or_inconclusive"),
                 theorem_status="candidate_pending_independent_proof_and_literature_review")
    completion(out,summary)
    return summary


def d2(out, full=False):
    start=time.monotonic(); out=freeze(out,"d2_full" if full else "d2_smoke")
    rows=[]; complete=True; reps=0; nesting_failures=0; nesting_unchecked=0
    grid=condition_grid()
    if not full: grid=[grid[0],grid[4],grid[8]]
    with (out/"replicates.jsonl").open("w") as stream:
        for condition in grid:
            reference=population_reference(condition)
            if full:
                first,count={"primary":(410001,1000),"secondary":(420001,200),"prevalence":(430001,200)}[condition["family"]]
                seeds=range(first,first+count)
            else: seeds=range(990001,990004)
            for index,seed in enumerate(seeds):
                if quota_exceeded(start,out): complete=False; break
                result=run_replicate(condition,seed,reference,nesting=(condition["family"]=="primary" and index<50))
                rows.extend(result["metrics"]); reps+=1
                nesting_failures+=len(result["shared_box"]["nesting_violations"]) if result["shared_box"] else 0
                nesting_unchecked+=len(result["shared_box"]["nesting_unchecked"]) if result["shared_box"] else 0
                stream.write(json.dumps(result,allow_nan=False)+"\n"); stream.flush()
            if not complete: break
    result=summarize(rows,smoke=not full)
    complete=complete and not quota_exceeded(start,out)
    result.update(complete=complete,replicates=reps,planned_full_replicates=13000,
                  nesting_failures=nesting_failures,nesting_unchecked=nesting_unchecked,elapsed_seconds=time.monotonic()-start)
    if not complete: result["coverage_decision"]="incomplete_protocol"
    completion(out,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("d1","d2"):
        p=sub.add_parser(name); p.add_argument("--output",required=True); p.add_argument("--full",action="store_true")
    gate=sub.add_parser("gate"); gate.add_argument("--input",required=True); gate.add_argument("--source",required=True); gate.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="gate":
        source=json.loads(Path(args.input).read_text()); out=freeze(args.output,"data_gate")
        result=validate_dataset(source["rows"],source["manifest"])
        actual_hash=hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
        result["source_hash_verified"]=actual_hash==source["manifest"].get("sha256","").lower()
        if not result["source_hash_verified"]:
            result.update(status="blocked_schema",coverage_enabled=False)
            result["reasons"].append("Actual source SHA256 does not match the manifest")
        write_json(out/"gate.json",result)
        completion(out,dict(complete=True,gate_status=result["status"],real_data_validation=False))
    else: result={"d1":d1,"d2":d2}[args.command](args.output,args.full)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__": main()
