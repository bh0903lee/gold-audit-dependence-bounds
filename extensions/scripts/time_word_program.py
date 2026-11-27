"""Computation time of one finite-sample interval (Algorithm 1 of the article, Section 3).

Draws observation and audit episodes from the main simulation design (stay probability 0.5, detection 0.95 after a latent
0 and 0.20 after a latent 1), N = 50,000 and B = 500, and times the full computation of one interval: the simultaneous
Clopper-Pearson moment intervals and the two linear programs (minimum and maximum) of the word program. Reports the median
over 30 repetitions per window length and audit information set, after one warm-up call, on one CPU thread. Times depend on
the machine; the processor is recorded in the output.

Usage (package root): OMP_NUM_THREADS=1 python -I extensions/scripts/time_word_program.py OUT.json
"""
import json, platform, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from topic_experiment.engine import bounds, sample_episodes  # noqa: E402


def cpu():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main(out):
    rng = np.random.default_rng(20261124)
    rows = []
    for L in (3, 4):
        for moments in ('marginal', 'neighbor', 'full'):
            times = []
            for r in range(31):
                _, y = sample_episodes(rng, 50000, L, 0.5, (0.95, 0.2))
                xa, ya = sample_episodes(rng, 500, L, 0.5, (0.95, 0.2))
                t = time.perf_counter()
                bounds(y, xa[:, 1], ya, moments=moments, interval='exact_binomial')
                if r:  # the first call is a warm-up
                    times.append(time.perf_counter() - t)
            rows.append({'L': L, 'moments': moments, 'repetitions': len(times), 'median_seconds': float(np.median(times)),
                         'max_seconds': float(np.max(times))})
    json.dump({'processor': cpu(), 'threads': 1, 'N': 50000, 'B': 500, 'interval': 'exact_binomial', 'rows': rows},
              open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
