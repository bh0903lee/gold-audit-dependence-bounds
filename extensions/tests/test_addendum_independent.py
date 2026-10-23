"""Independent checks of the extension runners: hand fixtures and adversarial recorded-data contracts.

No experiment seeds or saved results are opened here.
"""
import csv
import hashlib
import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
from c11_upgrade import e7
from c11_upgrade import e8


def manifest_for(path):
    return dict(dataset_name='room_occupancy_estimation',
        source_url='https://archive.ics.uci.edu/dataset/independent-synthetic-fixture',
        license='synthetic test fixture; not downloaded', gold_source='exact_synthetic_truth',
        units_evidence='two synthetic dates',
        selection_conditions=dict(public_license=True, independent_gold=True, fixed_detector=True, multiple_units=True),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), timestamp_column='timestamp',
        timestamp_format='iso', timezone='UTC', count_column='count', detector=dict(type='pir_or', columns=['pir']),
        lag_seconds=60, stationarity_rationale='synthetic contract test, no real stationarity claim', unit_column='unit')


def wframe(xs, ys, offset=0, day='2026-10-12'):
    return dict(x=list(xs), y=list(ys), unit='room', day=day,
                times=[float(offset),float(offset+60),float(offset+120)], block_id=1)


class IndependentE7Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='independent_fixture_')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'fixture.csv'

    def write(self, rows):
        with self.path.open('w', newline='') as f:
            writer=csv.writer(f); writer.writerow(['timestamp','unit','count','pir']); writer.writerows(rows)
        return manifest_for(self.path)

    def daily(self, minutes, values=None):
        rows=[]
        for day in (12,13):
            for index, minute in enumerate(minutes):
                count,pir=(1,1) if values is None else values(index)
                rows.append([(datetime(2026,10,day,9)+timedelta(minutes=minute)).isoformat(), 'room', count,pir])
        return rows

    def test_source_hash_and_candidate_rejection_are_before_inference(self):
        m=self.write(self.daily(range(6)))
        m['sha256']='0'*64
        with self.assertRaises(ValueError): e7.prepare_frame(self.path,m)
        m=manifest_for(self.path); m['source_url']='https://archive.ics.uci.edu.evil.invalid/a'
        with self.assertRaises(ValueError): e7.prepare_frame(self.path,m)
        m=manifest_for(self.path); m['dataset_name']='occupancy_detection'
        with self.assertRaises(ValueError): e7.prepare_frame(self.path,m)

    def test_stratum_requires_all_three_times_and_never_crosses_day(self):
        m=self.write(self.daily([-1,0,1,2,3,4,5,478,479,480]))
        result=e7.prepare_frame(self.path,m)
        self.assertEqual(len(result['frames']),8)
        self.assertEqual(result['unit_days'],2)
        for w in result['frames']:
            self.assertEqual(w['times'][1]-w['times'][0],60)
            self.assertEqual(w['times'][2]-w['times'][1],60)
            self.assertTrue(all(datetime.fromtimestamp(t).hour==9 for t in w['times']))
        self.assertEqual(result['excluded']['outside_fixed_stratum'],4)

    def test_invalid_middle_row_does_not_bridge_gap(self):
        m=self.write(self.daily(range(7),lambda i: ('nan',1) if i==3 else (1,1)))
        result=e7.prepare_frame(self.path,m,'block_200')
        self.assertEqual(len(result['frames']),4)
        self.assertEqual(len({w['block_id'] for w in result['frames']}),4)
        self.assertEqual(result['excluded']['missing_or_invalid_value'],2)

    def test_five_delta_uses_actual_lag_without_interpolation(self):
        m=self.write(self.daily(range(12)))
        result=e7.prepare_frame(self.path,m,lag_multiplier=5)
        self.assertEqual(len(result['frames']),4)
        self.assertEqual(len({w['block_id'] for w in result['frames']}),2)
        self.assertTrue(all(w['times'][1]-w['times'][0]==300 and w['times'][2]-w['times'][1]==300 for w in result['frames']))

    def test_fractional_native_lag_does_not_split_every_block(self):
        rows=[]
        for day in (12,13):
            start=datetime(2026,10,day,9)
            for i in range(10):
                rows.append([(start+timedelta(microseconds=100000*i)).isoformat(),'room',1,1])
        m=self.write(rows);m['lag_seconds']=.1
        result=e7.prepare_frame(self.path,m,'block_200')
        self.assertEqual(len(result['frames']),16)
        self.assertEqual(len({w['block_id'] for w in result['frames']}),2)
        gate=e7.gate_probe(result['frames'],m)
        self.assertEqual(gate['status'],'dependent_inference_required')

    def test_thinning_restarts_within_each_day(self):
        m=self.write(self.daily(range(111)))
        result=e7.prepare_frame(self.path,m,'thin_10')
        self.assertEqual(len(result['frames']),22)
        for day in ('2026-10-12','2026-10-13'):
            centers=[w['times'][1] for w in result['frames'] if w['day']==day]
            self.assertEqual(len(centers),11)
            self.assertEqual(np.diff(centers).tolist(),[600.]*10)

    def test_duplicate_after_probe_prefix_is_still_rejected(self):
        rows=self.daily(range(8)); rows.append(rows[-1])
        m=self.write(rows)
        with self.assertRaisesRegex(ValueError,'Duplicate'): e7.prepare_frame(self.path,m)

    def test_gate_keeps_dependency_and_is_explicitly_only_a_probe(self):
        m=self.write(self.daily(range(6)))
        f=e7.prepare_frame(self.path,m)
        gate=e7.gate_probe(f['frames'],m)
        self.assertEqual(gate['status'],'dependent_inference_required')
        self.assertFalse(gate['coverage_enabled'])
        self.assertFalse(gate['source_hash_verified'])
        self.assertEqual(gate['gate_scope'],'schema_contract_probe_not_full_cohort')

    def test_nonstationary_reference_0001_stays_undefined(self):
        frames=[wframe([0,0,0],[0,0,0]), wframe([0,0,1],[0,0,1],60)]
        ref=e7.empirical_reference(frames)
        self.assertAlmostEqual(ref['stationarity_residual'],.5)
        self.assertEqual(ref['theta'],1.) # first pairs are 00,00; second-pair mean would be 0.
        self.assertTrue(all(v['status']=='empty' for v in ref['intervals'].values()))

    def test_exact_degenerate_empirical_probability_does_not_exceed_one(self):
        ref=e7.empirical_reference([wframe([1,1,1],[1,1,1],60*i) for i in range(9)])
        self.assertEqual(ref['theta'],1.)
        self.assertTrue(all(v['status']=='ok' for v in ref['intervals'].values()))

    def test_fixed_threshold_training_is_separate_and_not_retrained(self):
        rows=self.daily(range(6),lambda i:(1,i))
        rows += [[datetime(2026,10,11,9).isoformat(),'room',1,999]]
        m=self.write(rows)
        m['detector']=dict(type='fixed_threshold',column='pir',threshold=2,direction='gt',
            training_start='2026-10-10T00:00:00',training_end='2026-10-11T23:59:59',
            evaluation_start='2026-10-12T00:00:00',training_rule='precommitted synthetic threshold',
            training_provenance='unit test fixture')
        f=e7.prepare_frame(self.path,m)
        self.assertEqual(f['excluded']['training_or_pre_evaluation'],1)
        self.assertEqual(f['frames'][0]['y'],[0,0,0])
        self.assertEqual(f['frames'][1]['y'],[0,0,1])
        m['detector']['training_end']='2026-10-12T09:00:00'
        with self.assertRaises(ValueError): e7.prepare_frame(self.path,m)

    def test_e7_disjoint_srs_uses_central_gold_and_independent_seed_recipe(self):
        frames=[]
        for i in range(120):
            frames.append(wframe([i%2,(i//2)%2,(i//3)%2],[i%2,(i//5)%2,(i//7)%2],60*i,
                                 '2026-10-12' if i<60 else '2026-10-13'))
        frame=dict(frames=frames,status='ready',lag_multiplier=1,method='application_only')
        reference=dict(theta=.25,f=.1,intervals={k:dict(status='ok',lower=-.5,upper=.5) for k in ('M','N','F')})
        captured={}
        def mock_solve(oy, ax, ay, epsilon, alpha):
            captured.update(oy=oy.copy(),ax=ax.copy(),ay=ay.copy(),epsilon=epsilon,alpha=alpha)
            return {k:dict(status='ok',lower=-1.,upper=1.) for k in ('M','N','F')}
        with patch.object(e7,'solve_information',mock_solve):
            result=e7.run_replicate(frame,50,990001,reference)
        generator=np.random.default_rng(np.random.SeedSequence([20261014,11,5,0,990001,1]))
        order=np.argsort(generator.random(120),kind='stable')
        self.assertEqual(result['audit_indices'],order[:50].tolist())
        self.assertEqual(result['calibration_indices'],order[50:100].tolist())
        self.assertFalse(set(result['audit_indices'])&set(result['calibration_indices']))
        self.assertEqual(captured['ax'].tolist(),[frames[i]['x'][1] for i in order[:50]])
        self.assertEqual(captured['ay'].tolist(),[frames[i]['y'] for i in order[:50]])
        self.assertEqual(captured['alpha'],.04)
        self.assertGreater(captured['epsilon'],0.)

    def test_small_finite_frame_is_not_sampled_with_replacement(self):
        frame=dict(frames=[wframe([0,0,0],[0,0,0])]*99,status='ready',lag_multiplier=1,method='application_only')
        with patch.object(e7,'solve_information',side_effect=AssertionError('Must not infer')):
            result=e7.run_replicate(frame,50,990001)
        self.assertEqual(result['status'],'insufficient_population')


class IndependentE8BlockTests(unittest.TestCase):
    def test_nonempty_block_weighting_and_hoeffding_match_hand_arithmetic(self):
        # Three blocks, one omitted audit block; first sampled block has 1 success/3,
        # last has 1/1. Equal nonempty mean=2/3, whereas pooled mean=1/2.
        oy=np.array([[0,0,0]]*8)
        idx=np.array([0,1,2,7]); ax=np.array([1,0,0,1]); blocks=np.array([0,0,0,1,1,1,2,2])
        limits,diag=e8.block_moment_limits(oy,ax,oy[idx],idx,blocks,.04,'N')
        self.assertAlmostEqual(diag['centers']['m'],2/3)
        self.assertEqual(diag['audit_block_counts'],[3,0,1])
        self.assertEqual(diag['nonempty_audit_blocks'],2)
        radius=math.sqrt(math.log(2/(.04/11))/(2*2))
        self.assertAlmostEqual(diag['radii']['m'],radius)
        self.assertEqual(limits['m'],(max(0,2/3-radius),min(1,2/3+radius)))
        upper,cal=e8.block_upper(1-ax,np.ones(4,dtype=int),idx,blocks)
        self.assertAlmostEqual(cal['center'],2/3)
        self.assertAlmostEqual(cal['radius'],math.sqrt(math.log(100)/(2*2)))
        self.assertEqual(upper,min(1,2/3+math.sqrt(math.log(100)/4)))

    def test_zero_calibration_counts_keep_positive_upper(self):
        upper,diag=e8.block_upper(np.ones(2,int),np.zeros(2,int),np.array([0,3]),np.array([0,0,1,1]))
        self.assertEqual(diag['center'],0.)
        self.assertGreater(upper,0.)
        self.assertFalse(diag['independence_guaranteed'])

    def test_priority_filters_before_selection_and_breaks_ties_by_index(self):
        p=np.array([.3,.01,.1,.3,.05,.3])
        self.assertEqual(e8.priority_select(p,np.array([0,3,5]),2).tolist(),[0,3])
        self.assertEqual(e8.priority_select(p,np.arange(6),2).tolist(),[1,4])

    def test_single_record_window_content_and_overlap(self):
        x=np.array([0,1,1,0,1]);y=np.array([1,0,1,1,0])
        wx,wy=e8.consecutive_windows(x,y)
        self.assertEqual(wx.tolist(),[[0,1,1],[1,1,0],[1,0,1]])
        self.assertEqual(wy.tolist(),[[1,0,1],[0,1,1],[1,1,0]])


class IndependentCLIContractTests(unittest.TestCase):
    def setUp(self):
        from c11_upgrade import addendum_cli
        self.cli=addendum_cli
        self.tmp=tempfile.TemporaryDirectory(prefix='independent_cli_')
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name).resolve()
        patcher=patch.object(addendum_cli,'OUTPUT_ROOT',root)
        patcher.start(); self.addCleanup(patcher.stop)
        self.base=root/'fixture'; self.base.mkdir()

    def test_output_guard_rejects_escape_nonempty_and_symlink(self):
        c=self.cli
        for path in (Path('/tmp/not-an-output-root-child'),c.OUTPUT_ROOT,c.OUTPUT_ROOT/'..'/'outside-child'):
            with self.assertRaises(ValueError): c.output_path(path)
        (self.base/'occupied').mkdir(); (self.base/'occupied'/'keep.txt').write_text('fixture')
        with self.assertRaises(ValueError): c.output_path(self.base/'occupied')
        (self.base/'escape').symlink_to('/tmp',target_is_directory=True)
        with self.assertRaises(ValueError): c.output_path(self.base/'escape'/'child')
        self.assertEqual(c.output_path(self.base/'new'),self.base/'new')

    def test_config_hash_mismatch_cannot_be_overridden(self):
        fake=self.base/'fake_source'; fake.mkdir(); (fake/'configs').mkdir(); (fake/'src').mkdir()
        (fake/'src'/'module.py').write_text('modified fixture')
        (fake/'configs/addendum_e7_e9.json').write_text(json.dumps(dict(baseline_hashes={'src/module.py':'0'*64})))
        with patch.object(self.cli,'ROOT',fake):
            with self.assertRaisesRegex(ValueError,'Protected baseline'): self.cli.validate_config()

    def test_invalid_selection_run_is_rejected_without_reaggregation(self):
        (self.base/'manifest.json').write_text(json.dumps(dict(kind='e8_full',run_complete=False)))
        with self.assertRaisesRegex(ValueError,'complete'): self.cli.e8_selection(self.base)
        with self.assertRaises(ValueError): self.cli.e8_selection(Path('/tmp/not-an-output-root-child'))

    def test_e8_selection_rejects_changed_reused_population_source(self):
        c=self.cli
        selected=dict(method='thin_10',eligible=['thin_10'],mean_widths={'thin_10':1.},
                      reason='eligible_minimum_width',coverage_certification=False)
        (self.base/'rows.jsonl').write_text('{}\n')
        (self.base/'summary.json').write_text(json.dumps(dict(selection=selected)))
        config=json.loads((c.ROOT/'configs/addendum_e7_e9.json').read_text())
        names=set(config['baseline_hashes'])|{'src/c11_upgrade/e8.py',
            'src/c11_upgrade/addendum_core.py','src/c11_upgrade/addendum_cli.py','configs/addendum_e7_e9.json'}
        hashes={name:c.sha256(c.ROOT/name) for name in names}
        hashes['src/c11_upgrade/population.py']='0'*64
        manifest=dict(kind='e8_full',run_complete=True,hashes=hashes,
                      output_hashes={name:c.sha256(self.base/name) for name in ('rows.jsonl','summary.json')})
        (self.base/'manifest.json').write_text(json.dumps(manifest))
        with patch.object(e8,'summarize',return_value=dict(complete_grid=True,selection=selected)):
            with self.assertRaisesRegex(ValueError,'different|source|baseline|implementation'):
                c.e8_selection(self.base)

    def test_static_plan_records_zero_executed_replicates(self):
        out=self.base/'static_plan'
        result=self.cli.plan(out)
        self.assertEqual(result['executed_replicates'],0)
        self.assertEqual(result['scientific_verdict'],'NOT_EVALUATED')
        manifest=json.loads((out/'manifest.json').read_text())
        self.assertFalse(manifest['scientific_run']); self.assertFalse(manifest['run_complete'])
        self.assertEqual(manifest['environment']['CUDA_VISIBLE_DEVICES'],'')
        self.assertIn('tests/test_addendum_independent.py',manifest['hashes'])


if __name__=='__main__': unittest.main()
