"""Strict data feasibility gate. Passing records never prove stationarity or IID."""
import math


def validate_dataset(rows, manifest):
    reasons=[]; statuses=[]
    def fail(status,reason): statuses.append(status); reasons.append(reason)
    required_manifest=("lag","binary_mapping","missing_policy","sampling_design","license","sha256","split_seed",
                       "independence_design","stationarity_rationale","population_id")
    if not isinstance(manifest,dict) or any(k not in manifest for k in required_manifest):
        return dict(status="blocked_schema",reasons=["Required manifest fields missing"],coverage_enabled=False)
    if not rows: return dict(status="blocked_schema",reasons=["No records"],coverage_enabled=False)
    if not isinstance(manifest["sha256"],str) or len(manifest["sha256"])!=64 or any(c not in "0123456789abcdef" for c in manifest["sha256"].lower()):
        fail("blocked_schema","Source SHA256 must have 64 hexadecimal characters")
    lag=manifest["lag"]
    if not isinstance(lag,(int,float)) or isinstance(lag,bool) or not math.isfinite(lag) or lag<=0:
        fail("blocked_schema","Lag must be finite and positive")
    if not manifest["stationarity_rationale"]:
        fail("stationarity_unsupported","No stationarity rationale; a test cannot certify the assumption")
    if manifest["independence_design"]!="independent_episodes":
        fail("dependent_inference_required","IID episode design is not documented")
    if manifest["sampling_design"]!="representative_probability_sample":
        fail("blocked_selection","Representative sampling is not documented")
    ids=set(); units={}; paired=False; roles=set(); calibration=[]; selection_by_cohort={}
    required=("episode_id","unit_id","t0","t1","t2","y0","y1","y2","x1_gold","gold_source",
              "cohort_role","selection_probability","independence_design","stationarity_rationale","population_id")
    for i,row in enumerate(rows):
        if not isinstance(row,dict) or any(k not in row for k in required):
            fail("blocked_schema",f"Row {i}: required fields missing"); continue
        if row["episode_id"] in ids: fail("blocked_schema",f"Row {i}: duplicate episode ID")
        ids.add(row["episode_id"])
        role=row["cohort_role"]; roles.add(role)
        if role not in ("obs","moment","calibration","reference"): fail("blocked_schema",f"Row {i}: unknown cohort role")
        if any(type(row[k]) not in (int,bool) or row[k] not in (0,1) for k in ("y0","y1","y2")):
            fail("blocked_schema",f"Row {i}: observations must be binary")
        if role!="obs" and (type(row["x1_gold"]) not in (int,bool) or row["x1_gold"] not in (0,1)):
            fail("blocked_gold",f"Row {i}: central gold missing or nonbinary")
        if role!="obs" and (not row["gold_source"] or row["gold_source"]=="classifier_proxy"):
            fail("blocked_gold",f"Row {i}: gold provenance is not validated truth")
        sp=row["selection_probability"]
        if not isinstance(sp,(int,float)) or isinstance(sp,bool) or not math.isfinite(sp) or not 0<sp<=1:
            fail("blocked_selection",f"Row {i}: selection probability must lie in (0,1]")
        else:
            selection_by_cohort.setdefault(role,[]).append(float(sp))
        if row["population_id"]!=manifest["population_id"] or not row["stationarity_rationale"]:
            fail("stationarity_unsupported",f"Row {i}: population transport/stationarity unsupported")
        if row["independence_design"]!="independent_episodes": fail("dependent_inference_required",f"Row {i}: dependence requires different inference")
        times=[row[k] for k in ("t0","t1","t2")]
        if not all(isinstance(t,(int,float)) and not isinstance(t,bool) and math.isfinite(t) for t in times):
            fail("blocked_schema",f"Row {i}: finite numeric times required")
        elif isinstance(lag,(int,float)) and (abs(times[1]-times[0]-lag)>1e-9 or abs(times[2]-times[1]-lag)>1e-9):
            fail("blocked_schema",f"Row {i}: incompatible lag")
        if all(isinstance(t,(int,float)) for t in times):
            for start,end in units.setdefault(row["unit_id"],[]):
                if max(start,times[0])<=min(end,times[2]): fail("dependent_inference_required",f"Row {i}: overlapping windows in a unit")
            units[row["unit_id"]].append((times[0],times[2]))
        adjacent=[row.get("x0_gold"),row.get("x2_gold")]
        if any(v is not None for v in adjacent):
            if role!="reference": fail("blocked_schema",f"Row {i}: adjacent truth must be in isolated reference cohort")
            if any(type(v) not in (int,bool) or v not in (0,1) for v in adjacent if v is not None): fail("blocked_gold",f"Row {i}: adjacent truth must be binary")
            else: paired=True
        if role=="calibration" and row["x1_gold"] in (0,1) and row["y1"] in (0,1):
            calibration.append(int(row["x1_gold"]==0 and row["y1"]==1))
    if not {"obs","moment","calibration"}.issubset(roles): fail("blocked_schema","obs, moment and calibration cohorts are required")
    for role,probabilities in selection_by_cohort.items():
        if max(probabilities)-min(probabilities)>1e-12:
            fail("blocked_selection",f"{role}: unequal inclusion probabilities require inference not implemented here")
    if manifest.get("self_weighting_sample") is not True:
        fail("blocked_selection","CP requires self-weighting IID sampling; IPW is not implemented")
    if "selection_support_complete" not in manifest or manifest["selection_support_complete"] is not True:
        fail("blocked_selection","Positive selection support must be documented for all population patterns")
    priority=("blocked_schema","blocked_gold","blocked_selection","dependent_inference_required","stationarity_unsupported")
    status=next((s for s in priority if s in statuses),"eligible_iid_validation" if paired else "application_only")
    out=dict(status=status,reasons=list(dict.fromkeys(reasons)),coverage_enabled=status=="eligible_iid_validation",
             source_hash_verified=False,
             assumption_caveat="Eligibility is conditional on documented assumptions; IID/stationarity are not certified.")
    if calibration and not statuses:
        from .inference import cp_upper
        out["epsilon_upper"]=cp_upper(sum(calibration),len(calibration))
        out["calibration_joint_fp_count"]=sum(calibration)
    return out


def fixture():
    manifest=dict(lag=1,binary_mapping={"0":0,"1":1},missing_policy="reject",sampling_design="representative_probability_sample",
                  license="synthetic fixture",sha256="0"*64,split_seed=20261014,independence_design="independent_episodes",
                  stationarity_rationale="Synthetic stationary law",population_id="synthetic",selection_support_complete=True,
                  self_weighting_sample=True)
    rows=[]
    for i,role in enumerate(("obs","moment","calibration","reference")):
        row=dict(episode_id=str(i),unit_id=str(i),t0=0,t1=1,t2=2,y0=0,y1=0,y2=1,
                 x1_gold=None if role=="obs" else 1,gold_source=None if role=="obs" else "exact_synthetic_latent",
                 cohort_role=role,selection_probability=1.,independence_design="independent_episodes",
                 stationarity_rationale="Synthetic stationary law",population_id="synthetic")
        if role=="reference": row.update(x0_gold=0,x2_gold=1)
        rows.append(row)
    return rows,manifest
