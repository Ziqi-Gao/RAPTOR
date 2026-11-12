import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import torch

from raptor.steering import gcav
from raptor.steering.controls import (atomic_json, checkpoint_valid, digest,
                                      install_observed_hooks, summarize_trajectories)
from raptor.steering.paper_judge import requests, aggregate, METHODS, METRICS
from raptor.steering.scalar_calibration import fit_mean_difference


class SteeringReleaseTests(unittest.TestCase):
    def test_raw_probe_exact_target_and_noop(self):
        w=torch.tensor([3.,4.],dtype=torch.float64);v=w/w.norm();hidden=torch.tensor([[1.,2.]],dtype=torch.float64)
        epsilon=gcav.compute_adaptive_epsilon(hidden,v,.3,.0001,probe_weight=w)
        probability=torch.sigmoid((hidden+epsilon*v)@w+.3).item()
        self.assertAlmostEqual(probability,.0001,places=12)
        self.assertEqual(gcav.compute_adaptive_epsilon(torch.tensor([[-100.,-100.]],dtype=torch.float64),v,.3,.0001,probe_weight=w).item(),0.)

    def test_validation_predicate_and_test_rejection(self):
        vectors=[np.ones(2)]*4
        self.assertEqual(gcav.select_active_layers(vectors,[.1,.7,.8,.1],accuracy_threshold=.7,reliability_enabled=True,reliability_comparison=">="),[1,2])
        self.assertEqual(gcav.select_active_layers(vectors,[.1,.7,.8,.1],accuracy_threshold=.7,reliability_enabled=True,reliability_comparison=">"),[2])
        with self.assertRaises(ValueError):
            gcav.register_steering_hooks_sequential(None,vectors,[1],test_accuracies=[1]*4)

    def test_observers_preserve_kernel_and_capture_denominator(self):
        def model():return SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])))
        arrays=dict(W=np.array([[3.,4.]]*3),B=np.array([.3]*3),directions=np.array([[.6,.8]]*3),val_acc_pre_refit=np.array([.8]*3))
        a,b=model(),model();vectors=[torch.tensor(row) for row in arrays['directions']];weights=[torch.tensor(row) for row in arrays['W']]
        native=gcav.register_steering_hooks_sequential(a,vectors,[1],target_prob=.0001,
                 probe_weights=weights,probe_biases=[.3]*3,reliability_enabled=False,accuracy_threshold=0,verbose=False)
        observed,trace=install_observed_hooks(b,arrays,[1],target=.0001,mode='raw_calibrated',activation_site='transformer_block_output')
        x=torch.tensor([[[1.,2.]]],dtype=torch.float64)
        self.assertTrue(torch.equal(a.model.layers[1](x),b.model.layers[1](x)))
        b.model.layers[1](x)
        for h in native+observed:h.remove()
        self.assertTrue(torch.equal(b.model.layers[1](x),x))
        self.assertEqual(len(trace),2)
        first=summarize_trajectories([dict(trajectory=trace)])
        all_steps=summarize_trajectories([dict(trajectory=trace)],'all_steps')
        self.assertEqual(first['N'],1);self.assertEqual(all_steps['N'],2)
        self.assertEqual(first['intervention_rate'],1.)

    def test_resume_rejects_corrupt_and_mismatched(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'unit.json';identity=dict(seed=42);payload=dict(text='example')
            atomic_json(p,dict(status='complete',identity=identity,payload=payload,payload_sha256=digest(payload)))
            self.assertTrue(checkpoint_valid(p,identity));self.assertFalse(checkpoint_valid(p,dict(seed=43)))
            value=json.loads(p.read_text());value['payload']['text']='changed';atomic_json(p,value)
            self.assertFalse(checkpoint_valid(p,identity))

    def test_scalar_calibration_native_identity(self):
        hidden=np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]])
        result=fit_mean_difference(hidden,np.array([0,0,1,1]),np.arange(4))
        self.assertTrue(result['converged']);self.assertEqual(result['epsilon'],1e-6)
        self.assertTrue(np.allclose(result['W'],result['a']*result['direction']))

    def test_original_judge_preserves_common_valid_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'run';generations=[]
            for seed in [1,2]:
                for method in METHODS:
                    generations.append(dict(model_id='test',task='STSA',direction='neutral_to_negative',prompt_id=0,generation_seed=seed,method=method,prompt_text='prompt',text='output'))
            requests(generations,run);mapping=json.loads((run/'local_records.json').read_text());scores=[]
            for row in mapping:
                scores.append(dict(record_id=row['record_id'],evaluable=True,valid=not(row['generation_seed']==2 and row['method']=='gcs'),reason='synthetic test',**{m:5 for m in METRICS}))
            scorefile=Path(d)/'scores.json';scorefile.write_text(json.dumps(scores))
            provenance=Path(d)/'provenance.json';provenance.write_text(json.dumps(dict(judge_model='synthetic',execution_context='unit test',scoring_date='test',rubric_description='test',source_kind='synthetic_test')))
            result=aggregate(run,scorefile,provenance)
            self.assertEqual(result['pairs'],2);self.assertEqual(result['common_valid_pairs'],1)
            scores[0]['target_sentiment_strength']=11;scorefile.write_text(json.dumps(scores))
            with self.assertRaises(ValueError):aggregate(run,scorefile,provenance)


if __name__=='__main__':unittest.main()
