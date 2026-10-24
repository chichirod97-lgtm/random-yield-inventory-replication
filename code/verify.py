import unittest,itertools,json,csv
from pathlib import Path
from dataclasses import replace
import numpy as np
from core import *

class Audit(unittest.TestCase):
    def test_arrival_and_terminal(self):
        c=Config(L=2,T=4,alpha=1,co=2,h=1,b=9)
        d=np.array([[1,2,3,4.]])
        result,rows=rollout(c,d,np.ones_like(d),lambda I,Q,t:np.full(len(I),5.),True)
        np.testing.assert_allclose([r['received'] for r in rows],[0,0,5,5])
        np.testing.assert_allclose([r['inventory_end'] for r in rows],[-1,-3,-1,0])
        np.testing.assert_allclose([r['order'] for r in rows],[5,5,0,0])
        self.assertEqual(result[0],65.)
    def test_failed_orders_not_paid(self):
        c=Config(L=2,T=4,alpha=1)
        v=rollout(c,np.zeros((1,4)),np.zeros((1,4)),lambda I,Q,t:10.)
        self.assertEqual(v[0],0.)
    def test_scenario_index_and_example(self):
        c=Config(L=3);B,w=scenarios(c)
        np.testing.assert_array_equal(B,[[0,0],[1,0],[0,1],[1,1]])
        self.assertAlmostEqual(w.sum(),1.)
        c=replace(c,p=.8);B,w=scenarios(c)
        self.assertAlmostEqual((np.maximum(100-30-B@np.array([40,20]),0)@w),22.)
    def test_reproducible_independent_splits(self):
        c=Config();a,b=generate(c,10,1);aa,bb=generate(c,10,1);x,z=generate(c,10,2)
        np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
        self.assertFalse(np.array_equal(a,x))
    def test_dp_matches_exhaustive_path_expectation(self):
        c=Config(L=1,T=4,p=.7,co=1,h=1,b=3,alpha=.95,mu=1,sigma=1)
        v,policy,_=finite_dp(c,[0,1],[.5,.5],[0,1,2])
        expected=0.
        for ds in itertools.product([0,1],repeat=c.T):
            for ys in itertools.product([0,1],repeat=c.T):
                prob=.5**c.T*np.prod([c.p if a else 1-c.p for a in ys])
                expected+=prob*rollout(c,np.array([ds]),np.array([ys]),policy)[0]
        self.assertAlmostEqual(v,expected,places=9)
    def test_grid(self): self.assertEqual(len(list(experiment_grid())),24)
    def test_discount(self):
        c=Config(T=4,L=2,alpha=.5)
        v=rollout(c,np.ones((1,4)),np.zeros((1,4)),lambda I,Q,t:0.)
        self.assertAlmostEqual(v[0],sum(.5**t*9*(t+1) for t in range(4)))

if __name__=='__main__': unittest.main(verbosity=2)
