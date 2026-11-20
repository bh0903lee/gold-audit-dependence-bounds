"""Regression tests for native-output silencing, CPU settings and binding checks; fixtures only."""
import contextlib
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from c11_followup import p2_cli
from c11_followup.p2_safety import require_cpu
from c11_followup.runtime import silence_numerical_output


class RuntimeRegression(unittest.TestCase):
    def test_native_streams_and_restoration(self):
        with tempfile.TemporaryFile() as capture:
            saved=[os.dup(fd) for fd in (1,2)]
            try:
                for fd in (1,2):os.dup2(capture.fileno(),fd)
                with silence_numerical_output():
                    print('forbidden')
                    os.write(1,b'forbidden');os.write(2,b'forbidden')
                os.write(1,b'restored')
            finally:
                for fd,old in zip((1,2),saved):os.dup2(old,fd);os.close(old)
            capture.seek(0);self.assertEqual(capture.read(),b'restored')

    def test_each_wrong_thread_setting_is_rejected(self):
        cpu={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with patch.dict(os.environ,cpu):
            require_cpu()
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
            with patch.dict(os.environ,dict(cpu,**{name:'8'})):
                with self.assertRaises(ValueError):require_cpu()
        with patch.dict(os.environ,dict(cpu,CUDA_VISIBLE_DEVICES='0')):
            with self.assertRaises(ValueError):require_cpu()

    def test_binding_omission_and_invalid_digest_fail_closed(self):
        cfg=p2_cli.load_config(p2_cli.CONFIG)
        for mode in ('empty','partial','invalid'):
            bad=copy.deepcopy(cfg);key=next(iter(bad['input_hashes']))
            if mode=='empty':bad['input_hashes']={}
            elif mode=='partial':del bad['input_hashes'][key]
            else:bad['input_hashes'][key]='invalid'
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'bad.json';path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):p2_cli.load_config(path)


if __name__=='__main__':unittest.main()
