"""Recorded occupancy series: data contract and descriptive finite-population evaluation.

No downloads, detector training, stationarization or automatic dataset substitution.
The episode-record checks of adapter.py are applied to a four-window probe and reported as a schema check only;
they do not establish independence.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
import numpy as np
from .adapter import validate_dataset
from .inference import cp_upper
from .population import X, Y, features, lp_interval
from .addendum_core import (rng, population_references, solve_information,
                           make_problem, interval_metrics)

BUDGETS = (50, 200, 500)
METHODS = ('application_only', 'thin_10', 'block_200')


def validate_manifest(source, manifest):
    """Check recorded assertions and actual bytes; assertions are not evidence proof."""
    required = ('dataset_name', 'source_url', 'license', 'gold_source', 'units_evidence',
                'selection_conditions', 'sha256', 'timestamp_column', 'timestamp_format',
                'timezone', 'count_column', 'detector', 'lag_seconds', 'stationarity_rationale')
    if not isinstance(manifest, dict) or any(k not in manifest for k in required):
        raise ValueError('Missing dataset-manifest fields')
    if manifest['dataset_name'] not in ('room_occupancy_estimation', 'occupancy_detection'):
        raise ValueError('Unsupported dataset (expected room_occupancy_estimation or occupancy_detection)')
    url = urlparse(manifest['source_url'])
    if url.scheme != 'https' or url.hostname != 'archive.ics.uci.edu':
        raise ValueError('Official HTTPS UCI source provenance required')
    for key in ('license', 'gold_source', 'units_evidence', 'stationarity_rationale'):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f'Missing provenance: {key}')
    if manifest['gold_source'] == 'classifier_proxy':
        raise ValueError('Classifier proxy is not recorded truth')
    if manifest['selection_conditions'] != dict(public_license=True, independent_gold=True,
                                                fixed_detector=True, multiple_units=True):
        raise ValueError('All metadata selection conditions must be documented')
    if manifest['dataset_name'] == 'occupancy_detection' and not manifest.get('first_candidate_ineligible_reason'):
        raise ValueError('Second candidate requires metadata rejection reason for first')
    hasher = hashlib.sha256()
    with Path(source).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != manifest['sha256']:
        raise ValueError('Source file SHA-256 does not match the manifest')
    lag = manifest['lag_seconds']
    if (isinstance(lag, bool) or not isinstance(lag, (int, float)) or not math.isfinite(lag)
            or lag <= 0 or round(lag*1e6) < 1 or not math.isclose(lag*1e6, round(lag*1e6), rel_tol=0, abs_tol=1e-6)):
        raise ValueError('Positive finite native lag at microsecond resolution required')
    ZoneInfo(manifest['timezone'])
    if not manifest.get('unit_column') and not manifest.get('unit_id'):
        raise ValueError('Physical unit column or fixed physical unit ID required')
    detector = manifest['detector']
    if not isinstance(detector, dict) or detector.get('type') not in ('pir_or', 'fixed_threshold'):
        raise ValueError('Unsupported detector type')
    if detector['type'] == 'pir_or':
        if not isinstance(detector.get('columns'), list) or not detector['columns']:
            raise ValueError('PIR sensor columns required')
    else:
        if (not detector.get('column') or detector.get('direction') not in ('gt', 'ge', 'lt', 'le')
                or type(detector.get('threshold')) not in (int, float)
                or not math.isfinite(detector['threshold'])):
            raise ValueError('Fixed sensor threshold contract invalid')
        for key in ('training_start', 'training_end', 'evaluation_start', 'training_rule', 'training_provenance'):
            if not detector.get(key):
                raise ValueError(f'Detector training provenance required: {key}')
        if not (_time(detector['training_start'], manifest) <= _time(detector['training_end'], manifest)
                < _time(detector['evaluation_start'], manifest)):
            raise ValueError('Training and evaluation periods must be disjoint and ordered')
    return digest


def _time(value, manifest):
    dt = (datetime.fromisoformat(value) if manifest['timestamp_format'] == 'iso'
          else datetime.strptime(value, manifest['timestamp_format']))
    zone = ZoneInfo(manifest['timezone'])
    if dt.tzinfo is None:
        # Ambiguous or nonexistent local wall times must have explicit UTC offsets.
        a, b = dt.replace(tzinfo=zone, fold=0), dt.replace(tzinfo=zone, fold=1)
        if a.utcoffset() != b.utcoffset():
            raise ValueError('Ambiguous/nonexistent local timestamp requires explicit offset')
        dt = a
    return dt.astimezone(zone)


def _ticks(dt):
    delta = dt.astimezone(timezone.utc) - datetime(1970,1,1,tzinfo=timezone.utc)
    return (delta.days*86400+delta.seconds)*1000000+delta.microseconds


def prepare_frame(source, manifest, method='application_only', lag_multiplier=1):
    if method not in METHODS or lag_multiplier not in (1, 5):
        raise ValueError('Unsupported method or lag multiplier')
    digest = validate_manifest(source, manifest)
    groups = defaultdict(dict)
    seen = set()
    excluded = defaultdict(int)
    detector = manifest['detector']
    eval_start = _time(detector['evaluation_start'], manifest) if detector['type'] == 'fixed_threshold' else None
    with Path(source).open(newline='', encoding=manifest.get('encoding', 'utf-8-sig')) as stream:
        reader = csv.DictReader(stream)
        columns = {manifest['timestamp_column'], manifest['count_column']}
        columns.update(detector['columns'] if detector['type'] == 'pir_or' else [detector['column']])
        if manifest.get('unit_column'): columns.add(manifest['unit_column'])
        if not columns.issubset(reader.fieldnames or []):
            raise ValueError('Source schema lacks declared columns')
        for row in reader:
            try:
                dt = _time(row[manifest['timestamp_column']], manifest)
            except (ValueError, TypeError):
                # Invalid timestamps are not silently assigned to neighboring times.
                raise ValueError('Invalid/ambiguous source timestamp') from None
            unit = str(row[manifest['unit_column']] if manifest.get('unit_column') else manifest['unit_id'])
            if not unit: raise ValueError('Missing physical unit')
            instant = dt.astimezone(timezone.utc)
            key = (unit, instant)
            if key in seen: raise ValueError('Duplicate physical unit/timestamp')
            seen.add(key)
            if eval_start is not None and instant < eval_start.astimezone(timezone.utc):
                excluded['training_or_pre_evaluation'] += 1; continue
            if dt.weekday() >= 5 or not 9 <= dt.hour < 17:
                excluded['outside_fixed_stratum'] += 1; continue
            try:
                count = float(row[manifest['count_column']])
                sensor = [float(row[c]) for c in (detector['columns'] if detector['type'] == 'pir_or' else [detector['column']])]
                if not all(math.isfinite(v) for v in [count, *sensor]) or count < 0 or not count.is_integer():
                    raise ValueError('Invalid count/sensor')
                x = int(count > 0)
                if detector['type'] == 'pir_or': y = int(any(v > 0 for v in sensor))
                else:
                    v, cut = sensor[0], detector['threshold']
                    y = int({'gt': v > cut, 'ge': v >= cut, 'lt': v < cut, 'le': v <= cut}[detector['direction']])
            except (ValueError, TypeError):
                excluded['missing_or_invalid_value'] += 1; continue
            groups[(unit, dt.date().isoformat())][instant] = (x, y)
    delta = timedelta(seconds=manifest['lag_seconds'] * lag_multiplier)
    frames = []; block_number = 0
    for (unit, day), points in sorted(groups.items()):
        windows = []
        for center in sorted(points):
            times = (center-delta, center, center+delta)
            if all(t in points for t in times):
                windows.append(dict(unit=unit, day=day, times=[t.timestamp() for t in times],
                                    time_us=[_ticks(t) for t in times],
                                    x=[points[t][0] for t in times], y=[points[t][1] for t in times]))
        if method == 'thin_10': windows = windows[::10]
        # A missing interval splits a block: no block silently bridges data gaps.
        last_time = None; within_block = 0
        for w in windows:
            step = round(manifest['lag_seconds']*1e6) * (10 if method == 'thin_10' else 1)
            if last_time is None or w['time_us'][1]-last_time != step or within_block == 200:
                block_number += 1; within_block = 0
            w['block_id'] = block_number; frames.append(w)
            within_block += 1; last_time = w['time_us'][1]
    days = len({(w['unit'], w['day']) for w in frames})
    return dict(frames=frames, method=method, lag_multiplier=lag_multiplier, source_sha256=digest,
                excluded=dict(excluded), unit_days=days,
                status='ready' if frames and days >= 2 else 'insufficient_population',
                inference_scope='finite_population_descriptive')


def empirical_reference(frames):
    if not frames: raise ValueError('No eligible windows')
    encode = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(X, Y))}
    counts = np.zeros((4, 4, 4), int)
    for w in frames: counts[tuple(encode[x, y] for x, y in zip(w['x'], w['y']))] += 1
    law = np.array([Fraction(int(v), len(frames)) for v in counts.ravel()], dtype=object).reshape(4,4,4)
    result = population_references(law)
    result['stationarity_residual'] = float(np.max(np.abs(features()[2] @ law.ravel())))
    result['theta_definition'] = 'first_pair_spin_on_treated_eligible_frame'
    x = np.array([w['x'][1] for w in frames]); y = np.array([w['y'][1] for w in frames])
    result['false_negative_rate'] = float(np.mean(y[x == 1] == 0)) if np.any(x == 1) else None
    return result


def gate_probe(frames, manifest, lag_multiplier=1):
    """Preserve actual times/units in a bounded schema probe.

    Full CSV value, time, duplicate and provenance validation is prepare_frame's
    responsibility. This small probe is NOT full-cohort validation or eligibility.
    """
    if len(frames) < 4:
        return dict(status='blocked_schema', reasons=['At least four eligible windows required'], coverage_enabled=False)
    rows = []
    origin = frames[0].get('time_us', [round(t*1e6) for t in frames[0]['times']])[0]
    for i, role in enumerate(('obs', 'moment', 'calibration', 'reference')):
        w = frames[i]
        ticks = w.get('time_us', [round(t*1e6) for t in w['times']])
        times = [(t-origin)/1e6 for t in ticks]
        row = dict(episode_id=f'contract-{i}', unit_id=w['unit'],
                   t0=times[0], t1=times[1], t2=times[2],
                   y0=w['y'][0], y1=w['y'][1], y2=w['y'][2],
                   x1_gold=None if role == 'obs' else w['x'][1],
                   gold_source=None if role == 'obs' else manifest['gold_source'],
                   cohort_role=role, selection_probability=1.,
                   independence_design='dependent_record_windows',
                   stationarity_rationale=manifest['stationarity_rationale'], population_id='frozen_e7_frame')
        if role == 'reference': row.update(x0_gold=w['x'][0], x2_gold=w['x'][2])
        rows.append(row)
    contract = dict(lag=manifest['lag_seconds']*lag_multiplier, binary_mapping={'X':'count>0','Y':manifest['detector']},
                    missing_policy='drop_invalid_then_require_exact_lag',
                    sampling_design='representative_probability_sample', license=manifest['license'],
                    sha256=manifest['sha256'], split_seed=20261014,
                    independence_design='dependent_record_windows',
                    stationarity_rationale=manifest['stationarity_rationale'], population_id='frozen_e7_frame',
                    self_weighting_sample=True, selection_support_complete=True)
    result = validate_dataset(rows, contract)
    result.update(probe_rows=len(rows), gate_scope='schema_contract_probe_not_full_cohort',
                  source_hash_verified=False, inference_scope='finite_population_descriptive')
    return result


def run_replicate(frame, budget, seed, reference=None):
    if (budget not in BUDGETS or type(seed) is not int or
            seed not in (*range(460001,461001),990001,990002,990003) or
            frame.get('method') not in METHODS or frame.get('lag_multiplier') not in (1,5)):
        raise ValueError('Budget or seed outside the registered ranges')
    windows = frame['frames']; n = len(windows)
    if frame['status'] != 'ready' or n < 2*budget:
        return dict(budget=budget, seed=seed, status='insufficient_population', metrics=[])
    design = 5 if frame['lag_multiplier'] == 1 else 6
    order = np.argsort(rng(design, BUDGETS.index(budget), seed, 1).random(n), kind='stable')
    audit, calibration = order[:budget], order[budget:2*budget]
    yy = np.asarray([w['y'] for w in windows], int)
    xx = np.asarray([w['x'][1] for w in windows], int)
    ref = empirical_reference(windows) if reference is None else reference
    diagnostics = {}
    if frame['method'] == 'block_200':
        from .e8 import block_moment_limits, block_upper
        block_ids = np.array([w['block_id'] for w in windows])
        epsilon, cal_diag = block_upper(xx[calibration], yy[calibration,1], calibration, block_ids)
        intervals = {}
        for info in ('M','N','F'):
            limits, diag = block_moment_limits(yy,xx[audit],yy[audit],audit,block_ids,.04,info)
            intervals[info] = lp_interval(make_problem(limits,epsilon,info)); diagnostics[info] = diag
        diagnostics['calibration'] = cal_diag
    else:
        epsilon = cp_upper(int(np.sum((xx[calibration] == 0) & (yy[calibration,1] == 1))), budget, .01)
        intervals = solve_information(yy,xx[audit],yy[audit],epsilon,.04)
    metrics = []
    for info, interval in intervals.items():
        metrics.append(dict(budget=budget,seed=seed,info=info,method=frame['method'],
                            lag_multiplier=frame['lag_multiplier'],epsilon=epsilon,
                            inference_scope='finite_population_descriptive',
                            **interval_metrics(interval,ref['intervals'][info],ref['theta'])))
    return dict(budget=budget,seed=seed,status='ok',metrics=metrics,diagnostics=diagnostics,
                audit_indices=audit.tolist(),calibration_indices=calibration.tolist())


def summarize(rows, frame, reference, gate, confirmatory=False):
    result = dict(inference_scope='finite_population_descriptive', scientific_verdict='NOT_EVALUATED',
                  frame_size=len(frame['frames']), unit_days=frame['unit_days'], method=frame['method'],
                  gate=gate, reference=reference, groups=[])
    groups = defaultdict(list)
    for row in rows: groups[(row['budget'],row['info'])].append(row)
    complete = True
    expected = set(range(460001,461001))
    for b in BUDGETS:
        for info in ('M','N','F'):
            g = groups.get((b,info),[])
            enough = frame['status'] == 'ready' and len(frame['frames']) >= 2*b
            if not enough:
                result['groups'].append(dict(budget=b,info=info,status='insufficient_population',n=0)); continue
            full = len(g)==1000 and {r['seed'] for r in g}==expected
            complete &= full
            if not g: continue
            ref_values = [r['contains_population'] for r in g if r['contains_population'] is not None]
            result['groups'].append(dict(budget=b,info=info,status='descriptive',n=len(g),
                reference_defined_n=len(ref_values),contains_population=None if not ref_values else float(np.mean(ref_values)),
                covers_truth=float(np.mean([r['covers_truth'] for r in g])),
                mean_width=float(np.mean([r['width'] for r in g])),
                empty_n=sum(r['empty_feasible'] for r in g),solver_failure_n=sum(r['solver_failure'] for r in g),
                fallback_n=sum(r['fallback_used'] for r in g)))
    result['complete_eligible_grid'] = bool(complete and any(len(frame['frames'])>=2*b for b in BUDGETS))
    means = {(g['budget'],g['info']):g.get('mean_width') for g in result['groups']}
    result['width_reductions'] = []
    for b in BUDGETS:
        for weak,strong in (('M','N'),('N','F')):
            a,z = means.get((b,weak)),means.get((b,strong))
            ratio = None if a is None or z is None or a==0 else 1-z/a
            result['width_reductions'].append(dict(budget=b,comparison=f'{strong}_vs_{weak}',value=ratio,
                undefined_reason=None if ratio is not None else 'missing_or_zero_denominator'))
    ratio = next(r['value'] for r in result['width_reductions'] if r['budget']==200 and r['comparison']=='N_vs_M')
    blocked = gate['status'].startswith('blocked')
    result['application_criterion'] = ('NOT_EVALUATED_test_fixture' if not confirmatory else
        'not_evaluable' if not result['complete_eligible_grid'] or blocked or ratio is None else
        'descriptive_criterion_met' if reference['f']<=.01 and ratio>=.10 else 'descriptive_criterion_not_met')
    return result
