import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from raptor.experiments.appendix import tasks, aggregate
from raptor.experiments.check_separability import svm_separability_check, perceptron_check
from raptor.experiments.structure_validation import plugin_acc_pred, scale_train_eval


class AppendixTests(unittest.TestCase):
    def test_training_only_transform(self):
        train=np.array([[-1.,0.],[1.,0.]])
        test=np.array([[100.,7.]])
        a,b=scale_train_eval(train,test,True)
        self.assertTrue(np.array_equal(a,train));self.assertTrue(np.array_equal(b,test))

    def test_plugin_known_probabilities(self):
        self.assertAlmostEqual(plugin_acc_pred(np.array([-1.,1.]),np.array([0.,1.]),1.,0.,0.),1.)
        self.assertAlmostEqual(plugin_acc_pred(np.array([-1.,1.]),np.array([0.,1.]),0.,0.,1.),.5)

    def test_separability_kernels(self):
        x=np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]])
        y=np.array([-1,-1,1,1])
        self.assertTrue(svm_separability_check(x,y,[1e4,1e6,1e8],0).separable)
        self.assertTrue(perceptron_check(x,y,2000,0).zero_error)

    def test_structure_plan_and_incomplete_grid(self):
        config=dict(models=['Qwen/Qwen2.5-7B-Instruct','meta-llama/Meta-Llama-3.1-8B-Instruct'],
            datasets=['STSA','common','hatexplain'],layers=[10,20],train_fracs=[.1,.2,.4,.6,.8,1],subsample_seeds=[0,1,2,3,4])
        with tempfile.TemporaryDirectory() as d:
            plan=tasks(config,Path(d)/'emb',Path(d)/'out')
            self.assertEqual(len(plan),360);self.assertEqual(len(set(r['output'] for r in plan)),360)
            with self.assertRaisesRegex(ValueError,'incomplete'):aggregate(plan,Path(d)/'out')


if __name__=='__main__':unittest.main()
