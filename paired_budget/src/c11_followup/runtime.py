"""Native-output silencing and SHA-256 bindings of the paired-comparison inputs."""
import contextlib
import ctypes
import hashlib
import os
from pathlib import Path
import re
import sys

@contextlib.contextmanager
def silence_numerical_output():
    """Discard Python streams and native stdout/stderr, including buffered C writes."""
    saved = []
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    libc = ctypes.CDLL(None)
    libc.fflush(None)
    try:
        with open(os.devnull, 'w') as sink:
            for fd in (1, 2):
                saved.append((fd, os.dup(fd)))
                os.dup2(sink.fileno(), fd)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                try:
                    yield
                finally:
                    sink.flush()
                    libc.fflush(None)
    finally:
        for fd, duplicate in reversed(saved):
            os.dup2(duplicate, fd)
            os.close(duplicate)


def verify_bindings(config, study):
    root = Path(__file__).resolve().parents[3]
    for group, mandatory in REQUIRED_BINDINGS[study].items():
        mapping = config.get(group)
        if not isinstance(mapping, dict) or set(mapping) != set(mandatory):
            raise ValueError('Missing or unexpected mandatory binding: ' + group)
        for relative, expected in mapping.items():
            if not isinstance(expected, str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
                raise ValueError('Invalid SHA-256 binding: ' + relative)
            path = (root / relative).resolve()
            if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Bound source or design changed: ' + relative)

REQUIRED_BINDINGS = {'p2': {'input_hashes': ('paired_budget/src/c11_followup/__init__.py',
                                         'paired_budget/src/c11_followup/p2.py',
                                         'paired_budget/src/c11_followup/p2_cli.py',
                                         'paired_budget/src/c11_followup/p2_safety.py',
                                         'paired_budget/src/c11_followup/runtime.py',
                                         'extensions/src/c11_upgrade/addendum_core.py',
                                         'extensions/src/c11_upgrade/e9.py',
                                         'extensions/src/c11_upgrade/inference.py',
                                         'extensions/src/c11_upgrade/population.py')}}
