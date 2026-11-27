"""Small utilities shared by the experiment modules."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def weights(logw):
    logw = np.asarray(logw, float)
    if not np.isfinite(logw).any():
        raise ValueError('All candidate likelihoods are nonfinite')
    return np.exp(logw - logsumexp(logw))

def kl(p, q):
    p, q = np.asarray(p), np.asarray(q)
    ix = p > 0
    if np.any(q[ix] <= 0):
        raise ValueError('KL has a support violation')
    return float(np.sum(p[ix] * np.log(p[ix] / q[ix])))

def stationary(P):
    if np.min(P) < 0 or not np.allclose(P.sum(1), 1):
        raise ValueError('Invalid transition matrix')
    n = len(P)
    pi = np.linalg.lstsq(np.vstack([P.T - np.eye(n), np.ones(n)]),
                           np.r_[np.zeros(n), 1.], rcond=None)[0]
    if np.min(pi) < -1e-10 or not np.allclose(pi @ P, pi):
        raise ValueError('Stationary solve failed')
    return np.maximum(pi, 0) / np.maximum(pi, 0).sum()

def rk4(f, x, dt):
    a=f(x); b=f(x+dt*a/2); c=f(x+dt*b/2); d=f(x+dt*c)
    return x + dt*(a+2*b+2*c+d)/6

def paired_summary(rows, metric, unit='seed'):
    """Aggregate within independent units before descriptive bootstrap."""
    groups = {}
    for r in rows:
        groups.setdefault(str(r[unit]), []).append(r[metric])
    vals=np.array([np.mean(v) for v in groups.values()])
    rng=np.random.default_rng(7331)
    interval=None if len(vals)<5 else np.quantile(
        vals[rng.integers(len(vals),size=(4000,len(vals)))].mean(1), [.025,.975]).tolist()
    return dict(mean=float(vals.mean()), units=len(vals), descriptive_interval=interval)


class AuditRows(list):
    """Append-only completed-row journal; partial failures are never silent deletions."""
    def __init__(self,out):
        super().__init__();self.path=Path(out)/'rows.partial.jsonl'
        if self.path.exists():raise ValueError('Partial journal already exists')
    def append(self,row):
        with self.path.open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        super().append(row)
