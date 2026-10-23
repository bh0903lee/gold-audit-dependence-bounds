"""CPU-only entry point for the paired comparison that replaces the learned false-positive budget by the true false-positive rate; the full run is a separate explicit command."""
import argparse
import contextlib
import io
import os
from collections import Counter
import json
from pathlib import Path
import resource
import time
import numpy as np
from . import p2
from .p2_safety import check_output, require_cpu, sha256, CpuBudget, REPOSITORY
from c11_upgrade.population import periodic_law
from c11_upgrade.addendum_core import population_references
from .runtime import verify_bindings

BASE = Path(__file__).resolve().parents[2]
CONFIG = BASE / 'configs/p2.json'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def load_config(path):
    config = json.loads(Path(path).read_text())
    verify_bindings(config, 'p2')
    for rel, expected in config['input_hashes'].items():
        if sha256(REPOSITORY / rel) != expected:
            raise ValueError('Bound input hash differs: '+rel)
    fixed = {'n_obs': 10000, 'budgets': [200, 500, 2000], 'law_seeds': [310041, 310080],
             'confirmatory_seeds': [480001, 480200], 'known_zero_law_seeds': [310068, 310080]}
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError('Configuration differs from the implemented design')
    if config['bootstrap']['replicates'] != 2000 or config['bootstrap']['quantile_method'] != 'linear':
        raise ValueError('Bootstrap settings differ from the design')
    for rel, expected in config.get('new_code_hashes', {}).items():
        if sha256(BASE / rel) != expected:
            raise ValueError('Code hash differs: '+rel)
    return config


def plan():
    return {'protocol_id': 'paired-fp-budget-comparison', 'laws': 40, 'budgets': list(p2.BUDGETS),
        'replicates': 200, 'sample_intervals': 291600, 'sample_endpoint_solves': 583200,
        'auxiliary_population_intervals': 144000, 'fixed_reference_intervals': 120,
        'total_intervals_without_retries': 435720,
        'confirmatory_execution_performed': False}


@contextlib.contextmanager
def _silence_native_output():
    """Suppress native solver writes to fd 1/2 as well as Python stream redirects."""
    saved = []
    try:
        with open(os.devnull, 'w') as sink:
            for fd in (1, 2):
                saved.append((fd, os.dup(fd)))
                os.dup2(sink.fileno(), fd)
            yield
    finally:
        for fd, duplicate in reversed(saved):
            os.dup2(duplicate, fd)
            os.close(duplicate)


def smoke(output, quota):
    """Timing-only isolation: suppress downstream diagnostics and sanitize exceptions."""
    with _silence_native_output(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            return _smoke(output, quota)
        except Exception as error:
            raise RuntimeError('Timing run failed; error_type=' + type(error).__name__) from None


def _smoke(output, quota):
    """Do not serialize or print intervals, widths, losses or coverage."""
    status = Counter(); cpu_start = time.process_time(); wall_start = time.perf_counter()
    reference_cpu = 0.; replicate_cpu = 0.; intervals = 0; measured = []
    for index, budget in ((0, 200), (27, 500), (39, 2000)):
        spec = p2.law_specs()[index]
        start = time.process_time()
        law = periodic_law(spec['law_seed'], 3, 4)
        reference = population_references(law)
        reference_cpu += time.process_time()-start
        for seed in p2.FIXTURE_SEEDS:
            quota.check(); start = time.process_time()
            rows = p2.run_replicate(spec, budget, seed, law=law, reference=reference)
            elapsed = time.process_time()-start
            replicate_cpu += elapsed
            count = len(rows)+6  # 2 varying-epsilon auxiliary references × 3 information sets.
            intervals += count
            status.update(r['status'] for r in rows)
            measured.append({'law_index': index, 'budget': budget, 'fixture_seed': seed,
                'cpu_seconds': elapsed, 'interval_operations': count})
            del rows
    # Aggregate overhead measured with deterministic synthetic widths, not experiment outcomes.
    start = time.process_time()
    fixture = np.random.default_rng(990010).uniform(.4, 1.8, size=(40, 200, 4, 3))
    p2.paired_bootstrap(fixture, 2000)
    bootstrap_cpu = time.process_time()-start
    unit = replicate_cpu/intervals
    estimate = unit*(291600+144000)+(reference_cpu/3)*40+bootstrap_cpu
    result = {**plan(), 'mode': 'fixture_timing_only', 'measured_replicates': 9,
        'timings': measured, 'unit_cpu_seconds_per_interval_including_sampling': unit,
        'reference_generation_and_solve_cpu_seconds_3_laws': reference_cpu,
        'bootstrap_cpu_seconds': bootstrap_cpu, 'estimated_full_cpu_seconds': estimate,
        'estimated_full_cpu_seconds_safety_factor_2': 2*estimate,
        'observed_cpu_seconds': time.process_time()-cpu_start,
        'observed_wall_seconds': time.perf_counter()-wall_start,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'solver_status_counts': dict(status),
        'limitations': ['Nine fixture replicates; full run memory not measured',
                       'LP retries included in observed time, workload variation unresolved',
                       'Full raw-row aggregation memory upper bound requires execution monitoring'],
        'scientific_values_recorded': False}
    write_json(output/'resource_estimate.json', result)
    return result


def full(output, quota, config):
    rows = []; start = time.process_time()
    state = {'status': 'running', 'confirmatory': True, 'complete_replicates': 0}
    write_json(output/'run_state.json', state)
    try:
        with (output/'rows.jsonl').open('x') as fh:
            for spec in p2.law_specs():
                quota.check()
                reference = population_references(periodic_law(spec['law_seed'], 3, 4))
                for budget in p2.BUDGETS:
                    for seed in p2.CONFIRM_SEEDS:
                        quota.check()
                        batch = p2.run_replicate(spec, budget, seed, confirmatory=True, reference=reference)
                        for row in batch:
                            fh.write(json.dumps(row, allow_nan=False)+'\n')
                        fh.flush(); rows.extend(batch)
                        state['complete_replicates'] += 1
                        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > config['resources']['memory_limit_kib']:
                            raise MemoryError('Memory quota reached; partial run retained')
                        write_json(output/'run_state.json', state)
        quota.check()
        summary = p2.summarize(rows, confirmatory=True,
            bootstrap_reps=config['bootstrap']['replicates'], bootstrap_seed=config['bootstrap']['seed'])
        write_json(output/'summary.json', summary)
        state.update(status='complete', cpu_seconds=time.process_time()-start)
    except (TimeoutError, MemoryError) as error:
        state.update(status='partial_quota_stop', h1='not_evaluable', reason=str(error),
                     cpu_seconds=time.process_time()-start)
    except Exception as error:
        state.update(status='failed', h1='not_evaluable', reason=type(error).__name__+': '+str(error),
                     cpu_seconds=time.process_time()-start)
        raise
    finally:
        write_json(output/'run_state.json', state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'p2-smoke', 'p2-full'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=str(CONFIG))
    parser.add_argument('--max-cpu-seconds', type=float, default=1500)
    args = parser.parse_args(argv)
    require_cpu(); config = load_config(args.config)
    output = check_output(args.output, create=True)
    hashes = {str(p.relative_to(BASE)):sha256(p) for p in sorted(BASE.rglob('p2*.py'))}
    write_json(output/'manifest.json', {'config': config, 'config_sha256': sha256(args.config),
        'new_code_sha256': hashes, 'command': args.command, 'output_resolved': str(output),
        'cpu_limit_seconds': args.max_cpu_seconds, 'cpu_only': True})
    if args.command == 'plan':
        write_json(output/'plan.json', plan()); print('Plan written; no law generation or experiment.')
        return
    quota = CpuBudget(args.max_cpu_seconds)
    if args.command == 'p2-smoke':
        smoke(output, quota); print('Fixture timing completed; scientific values not persisted.')
    else:
        full(output, quota, config); print('Execution ended; see run_state.json for completion status.')


if __name__ == '__main__':
    main()
