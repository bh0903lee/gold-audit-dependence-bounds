"""Shared inference for the overlapping-window experiment, the general-law experiment and the recorded-occupancy application.

The marginal (M), neighbor (N) and full (F) audits each split the moment error budget over their own constraint
count (9, 11, 24), so finite-sample intervals need not be nested."""
import numpy as np
from .population import Problem, features, law_limits, lp_interval
from .inference import sample_counts, cp_interval

INFORMATION = ("M", "N", "F")
CONSTRAINT_COUNTS = {"M": 9, "N": 11, "F": 24}
TOLERANCE = 1e-8


def rng(design, cid, seed, cohort):
    values = [design, cid, seed, cohort]
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 0 for v in values):
        raise ValueError("Nonnegative integer stream identifiers required")
    return np.random.default_rng(np.random.SeedSequence([20261014, 11, *values]))


def information_names(info):
    if info not in INFORMATION:
        raise ValueError("Information must be M, N, or F")
    audits = {"M": ("m",), "N": ("m", "a", "b"),
              "F": tuple(f"u{k}" for k in range(16))}
    return tuple(f"o{k}" for k in range(8)) + audits[info]


def information_limits(obs_y, audit_x1, audit_y, info="N", alpha=.04):
    names = information_names(info)
    oc, ac = sample_counts(obs_y), sample_counts(audit_y, audit_x1)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Moment error budget must be in (0,1)")
    delta = alpha / CONSTRAINT_COUNTS[info]
    return {key: cp_interval(oc[key] if key.startswith("o") else ac[key],
                            len(obs_y) if key.startswith("o") else len(audit_y), delta)
            for key in names}


def direct_moment_limits(obs_y, audit_x1, audit_y, alpha=.04, info="N"):
    return information_limits(obs_y, audit_x1, audit_y, info, alpha)


def make_problem(limits, epsilon, info="N"):
    """Build M as well as N/F without changing population.INFO."""
    names = information_names(info)
    if not np.isfinite(float(epsilon)) or not 0 <= float(epsilon) <= 1:
        raise ValueError("epsilon must be a joint probability")
    _, rows, flow, c = features(3, 4)
    ae, be = [np.ones(len(c), int), *flow], [1, *([0] * len(flow))]
    au, bu = [rows["fp"]], [epsilon]
    for name in names:
        lo, hi = limits[name]
        if not all(np.isfinite(float(v)) for v in (lo, hi)) or not 0 <= lo <= hi <= 1:
            raise ValueError(f"Invalid probability interval: {name}")
        if lo == hi:
            ae.append(rows[name]); be.append(lo)
        else:
            au.extend([rows[name], -rows[name]]); bu.extend([hi, -lo])
    return Problem(c.copy(), np.asarray(ae), np.asarray(be, dtype=object),
                   np.asarray(au), np.asarray(bu, dtype=object), names)


def validate_law(law):
    p = np.asarray(law, dtype=object)
    if p.shape not in ((4, 4, 4), (64,)):
        raise ValueError("A four-state triple law is required")
    numeric = np.asarray(p, float).reshape(-1)
    if not np.isfinite(numeric).all() or np.any(numeric < 0) or abs(float(numeric.sum()) - 1) > 1e-10:
        raise ValueError("Law must be a normalized nonnegative distribution")
    return p.reshape(4, 4, 4)


def population_references(law):
    p = validate_law(law)
    _, rows, flow, objective = features()
    flat = p.reshape(-1)
    f = rows["fp"] @ flat
    limits = law_limits(p)
    intervals = {info: lp_interval(make_problem(limits, f, info)) for info in INFORMATION}
    return {"theta": float(objective @ flat), "f": float(f), "intervals": intervals,
            "stationarity_residual": float(np.max(np.abs(np.asarray(flow @ flat, float))))}


def solve_information(obs_y, audit_x1, audit_y, epsilon, alpha=.04):
    return {info: lp_interval(make_problem(information_limits(obs_y, audit_x1, audit_y, info, alpha),
                                           epsilon, info)) for info in INFORMATION}


def interval_metrics(interval, reference, theta):
    """Keep raw miss, empty fallback, solver unknown, and undefined references distinct."""
    status, reference_status = interval["status"], reference["status"]
    allowed = ("ok", "empty", "solver_failure")
    if status not in allowed or reference_status not in allowed:
        raise ValueError("Unrecognized LP status")
    if not np.isfinite(theta) or not -1 - TOLERANCE <= theta <= 1 + TOLERANCE:
        raise ValueError("Invalid theta")
    for item in (interval, reference):
        if item["status"] == "ok":
            lo, hi = item["lower"], item["upper"]
            if not all(np.isfinite(v) for v in (lo, hi)) or not -1-TOLERANCE <= lo <= hi+TOLERANCE <= 1+2*TOLERANCE:
                raise ValueError("Malformed successful interval")
    valid = status == "ok"
    defined = reference_status == "ok"
    empty, failure = status == "empty", status == "solver_failure"
    lo, hi = (interval["lower"], interval["upper"]) if valid else (-1., 1.)
    contains = (valid and lo <= reference["lower"] + TOLERANCE and hi >= reference["upper"] - TOLERANCE) if defined else None
    truth = bool(valid and lo-TOLERANCE <= theta <= hi+TOLERANCE)
    return {"status": status, "lower": interval.get("lower"), "upper": interval.get("upper"),
            "width": float(max(0., hi-lo)), "contains_population": None if contains is None else bool(contains),
            "covers_truth": truth, "population_reference_status": reference_status,
            "population_reference_defined": defined,
            "population_lower": reference.get("lower") if defined else None,
            "population_upper": reference.get("upper") if defined else None,
            "fallback_contains_population": None if failure or not defined else bool(contains or empty),
            "fallback_covers_truth": None if failure else bool(truth or empty),
            "empty_feasible": empty, "solver_failure": failure, "fallback_used": empty or failure,
            "theta": float(theta), "residual": interval.get("residual")}
