"""General-law experiment: finite-sample M/N/F intervals over 80 stationary laws (40 three-state laws with zero false-positive mass, 40 four-state laws with a learned false-positive budget). Laws are generated only inside run_replicate."""
import hashlib
import numpy as np
from .population import periodic_law, features, X, Y
from .inference import cp_upper, cp_interval
from .addendum_core import (INFORMATION, rng, validate_law, population_references,
                            solve_information, interval_metrics)

BUDGETS = (200, 500, 2000)
CONFIRM_SEEDS = range(450001, 450201)
TEST_SEEDS = range(990001, 990004)


def law_specs():
    """Metadata only: no call to periodic_law and no result access."""
    for bank, start, alphabet, offset in (("A", 300001, 3, 0), ("C", 310041, 4, 40)):
        for j in range(40):
            yield {"bank": bank, "law_seed": start+j, "alphabet": alphabet, "law_index": offset+j}


def _approved_spec(spec):
    if not isinstance(spec, dict) or not isinstance(spec.get("law_index"), int):
        raise ValueError("A law specification with integer index is required")
    index = spec["law_index"]
    approved = list(law_specs())
    if not 0 <= index < len(approved) or any(spec.get(k) != v for k, v in approved[index].items()):
        raise ValueError("Unregistered law specification (including the excluded four-state law seeds)")
    return approved[index]


def sample_words(generator, law, n):
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
        raise ValueError("Positive integer sample size required")
    p = np.asarray(validate_law(law), float).reshape(-1)
    # Only remove float conversion roundoff; the input law was already validated.
    p = p / p.sum()
    words = features()[0][generator.choice(64, size=n, p=p)]
    return X[words], Y[words]


def _sample_hash(*arrays):
    digest = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a, dtype="<i8")
        digest.update(str(a.shape).encode("ascii")); digest.update(a.tobytes())
    return digest.hexdigest()


def run_replicate(spec, budget, seed, *, confirmatory=False, law=None, reference=None, n_obs=10000):
    """One replicate; only confirmatory=True generates the registered laws.

    Tests must supply a separate law fixture and seed 990001--990003. The caller
    owns persistence and quota checks; this pure routine writes no output.
    """
    spec = _approved_spec(spec)
    if budget not in BUDGETS:
        raise ValueError("Budget must be 200, 500 or 2000")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("Integer replicate seed required")
    if confirmatory:
        if seed not in CONFIRM_SEEDS or law is not None or n_obs != 10000:
            raise ValueError("Experiment mode requires seeds 450001-450200, a generated law and n_obs=10000")
        law = periodic_law(spec["law_seed"], alphabet=spec["alphabet"])
    elif seed not in TEST_SEEDS or law is None:
        raise ValueError("Fixture mode requires an explicit law and a nonconfirmatory test seed")
    law = validate_law(law)
    cid = 3 * spec["law_index"] + BUDGETS.index(budget)
    ref = population_references(law) if reference is None else reference
    if spec["bank"] == "A" and ref["f"] > 1e-12:
        raise ValueError("Three-state (known-zero) laws must have zero false-positive mass")
    _, oy = sample_words(rng(4, cid, seed, 0), law, n_obs)
    ax, ay = sample_words(rng(4, cid, seed, 1), law, budget)
    calibration_n, calibration_k, epsilon = 0, None, 0.
    hashes = {"observation": _sample_hash(oy), "audit": _sample_hash(ax[:, 1], ay)}
    if spec["bank"] == "C":
        cx, cy = sample_words(rng(4, cid, seed, 2), law, budget)
        calibration_n = budget
        calibration_k = int(np.sum((cx[:, 1] == 0) & (cy[:, 1] == 1)))
        epsilon = cp_upper(calibration_k, calibration_n, .01)
        hashes["calibration"] = _sample_hash(cx[:, 1], cy[:, 1])
    intervals = solve_information(oy, ax[:, 1], ay, epsilon, .05 if spec["bank"] == "A" else .04)
    common = {**spec, "condition_id": cid, "budget": budget, "seed": int(seed),
              "confirmatory": bool(confirmatory), "n_obs": int(n_obs), "audit_n": budget,
              "epsilon": float(epsilon), "calibration_n": calibration_n, "calibration_k": calibration_k,
              "calibration_policy": "known_zero" if spec["bank"] == "A" else "learned_upper",
              "true_f": ref["f"], "f_zero": bool(abs(ref["f"]) <= 1e-12), "sample_hashes": hashes}
    metrics = []
    for info in INFORMATION:
        pop = ref["intervals"][info]
        metrics.append({**common, "method": info,
                        **interval_metrics(intervals[info], pop, ref["theta"]),
                        "population_zero_width": bool(pop["upper"]-pop["lower"] <= 1e-8) if pop["status"] == "ok" else None})
    return {"spec": spec, "budget": budget, "seed": int(seed), "condition_id": cid,
            "reference": ref, "sample_hashes": hashes, "metrics": metrics}


def _quantiles(values):
    return {"n": len(values), "p10": None, "median": None, "p90": None} if not values else dict(
        zip(("p10", "median", "p90"), np.quantile(values, [.1, .5, .9], method="linear").tolist()), n=len(values))


def summarize(rows, confirmatory=False):
    """Check grid completeness before the descriptive comparisons; incomplete laws are reported, never dropped."""
    rows = list(rows)
    groups = {}
    malformed = 0
    for row in rows:
        try:
            spec = _approved_spec(row)
            budget = row["budget"]
            if budget not in BUDGETS or row["method"] not in INFORMATION:
                raise ValueError("Invalid budget or method")
            if row["condition_id"] != 3*spec["law_index"] + BUDGETS.index(budget):
                raise ValueError("Wrong condition id")
            groups.setdefault((spec["law_index"], budget, row["method"]), []).append(row)
        except (KeyError, ValueError, TypeError):
            malformed += 1
    expected = set(CONFIRM_SEEDS)
    def complete(group):
        return (len(group) == 200 and {r["seed"] for r in group} == expected
                and all(r.get("confirmatory") is True and r.get("n_obs") == 10000
                        and r.get("audit_n") == r["budget"]
                        and r.get("calibration_n") == (r["budget"] if r["bank"] == "C" else 0)
                        for r in group))
    full = malformed == 0 and len(groups) == 80*3*3 and all(complete(g) for g in groups.values())
    summary_groups = []
    by_key = {}
    for (index, budget, info), group in sorted(groups.items()):
        n = len(group)
        defined = [r for r in group if r.get("population_reference_defined") is True]
        hits = sum(r.get("contains_population") is True for r in defined)
        mc = list(cp_interval(hits, len(defined))) if defined else None
        widths = [r["width"] for r in group]
        finite_widths = all(isinstance(v, (int, float)) and np.isfinite(v) and 0 <= v <= 2+1e-8 for v in widths)
        failure_n = sum(bool(r.get("solver_failure")) for r in group)
        entry = {"law_index": index, "bank": group[0]["bank"], "law_seed": group[0]["law_seed"],
                 "budget": budget, "method": info, "n": n, "reference_defined_n": len(defined),
                 "reference_undefined_n": n-len(defined), "contains_population_hits": hits,
                 "contains_population": hits/len(defined) if defined else None,
                 "contains_population_total_denominator": hits/n,
                 "mc_95": mc, "covers_truth": sum(r.get("covers_truth") is True for r in group)/n,
                 "empty_n": sum(bool(r.get("empty_feasible")) for r in group),
                 "solver_failure_n": failure_n, "fallback_n": sum(bool(r.get("fallback_used")) for r in group),
                 "mean_width": float(np.mean(widths)) if finite_widths else None,
                 "f_zero": group[0].get("f_zero"), "population_zero_width": group[0].get("population_zero_width"),
                 "complete": complete(group), "h9a_lower_ge_093": bool(mc[0] >= .93) if mc and len(defined)==n else None}
        summary_groups.append(entry); by_key[index, budget, info] = entry
    comparisons = []
    for spec in law_specs():
        for budget in BUDGETS:
            triple = [by_key.get((spec["law_index"], budget, info)) for info in INFORMATION]
            record = {**spec, "budget": budget}
            for weak, strong, label in ((0, 1, "N_vs_M"), (1, 2, "F_vs_N")):
                a, b = triple[weak], triple[strong]
                reason = ("missing_group" if a is None or b is None else
                          "invalid_width" if a["mean_width"] is None or b["mean_width"] is None else
                          "zero_denominator" if a["mean_width"] == 0 else None)
                ratio = None if reason else 1-b["mean_width"]/a["mean_width"]
                comparable = bool(reason is None and a["reference_undefined_n"] == 0 and b["reference_undefined_n"] == 0)
                record[label] = {"relative_reduction": ratio, "undefined_reason": reason,
                                 "reference_comparable": comparable,
                                 "strictly_narrower": bool(comparable and b["mean_width"] < a["mean_width"]-1e-8)}
            comparisons.append(record)
    at500 = [r for r in comparisons if r["budget"] == 500]
    successes = sum(r["N_vs_M"]["strictly_narrower"] for r in at500)
    comparable = sum(r["N_vs_M"]["reference_comparable"] for r in at500)
    all_references_defined = bool(full and all(r["reference_undefined_n"] == 0 for r in summary_groups))
    verdict = ("NOT_EVALUATED_fixture_only" if not confirmatory else
               "BLOCKED_incomplete_grid" if not full else
               "BLOCKED_undefined_population_reference" if not all_references_defined else
               "prediction_consistent" if successes >= 60 else "prediction_not_met")
    quantiles = {}
    for budget in BUDGETS:
        for bank in ("all", "A", "C"):
            selected = [r for r in comparisons if r["budget"] == budget and (bank == "all" or r["bank"] == bank)]
            quantiles[f"B{budget}_{bank}"] = {label: _quantiles([r[label]["relative_reduction"] for r in selected
                                                              if r[label]["relative_reduction"] is not None])
                                             for label in ("N_vs_M", "F_vs_N")}
    return {"confirmatory_requested": bool(confirmatory), "full_grid_complete": full,
            "malformed_rows": malformed, "row_count": len(rows), "groups": summary_groups,
            "all_population_references_defined": all_references_defined,
            "comparisons": comparisons, "relative_reduction_quantiles": quantiles,
            "h9a": "descriptive_only_no_simultaneous_guarantee" if confirmatory and full and all_references_defined else "NOT_EVALUATED",
            "h9a_below_threshold": [r for r in summary_groups if r["method"] == "N" and r["h9a_lower_ge_093"] is False],
            "h9a_undefined": [r for r in summary_groups if r["method"] == "N" and r["h9a_lower_ge_093"] is None],
            "h9b": verdict, "h9b_success_laws": successes, "h9b_total_laws": 80,
            "h9b_comparable_laws": comparable}
