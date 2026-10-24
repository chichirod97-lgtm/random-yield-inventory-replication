import unittest,tempfile
import numpy as np
import torch
from core import *
from learning import Weights,costs
from learning_v2 import *
class NumericalAudit(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_same_initial_parameters_and_policies(self):
        c=Config(T=6);torch.manual_seed(3);a=Weights(c);torch.manual_seed(3);b=DiagnosticWeights(c)
        for x,y in zip(a.parameters(),b.parameters()):np.testing.assert_array_equal(x.detach().numpy().astype('float64'),y.detach().numpy())
        I=np.array([-20.,0,20]);Q=np.array([[10.],[15.],[0.]])
        with torch.no_grad():
            x=a(torch.tensor(I,dtype=torch.float32),torch.tensor(Q,dtype=torch.float32),0)
            y=b(torch.tensor(I),torch.tensor(Q),0)
        np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-5)
    def test_teacher_initialization_matches_wnh(self):
        for L in [2,5]:
            c=Config(L=L);torch.manual_seed(3);m=DiagnosticWeights(c,'teacher_scaled64')
            I=np.array([-50.,0,40]);Q=np.full((3,L-1),10.)
            with torch.no_grad():v=m(torch.tensor(I),torch.tensor(Q),0).numpy()
            np.testing.assert_allclose(v,heuristic(c,'WNH')(I,Q,0),rtol=1e-7)
    def test_adam_second_moment_failure_detected(self):
        m=torch.nn.Linear(1,1,dtype=torch.float32);o=torch.optim.Adam(m.parameters())
        for p in m.parameters():p.grad=torch.full_like(p,1e30)
        o.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))
        with self.assertRaisesRegex(FloatingPointError,'optimizer'):check_numerics(m,o,'test')
    def test_float64_exponential_and_adam_are_finite(self):
        p=torch.nn.Parameter(torch.tensor(109.,dtype=torch.float64));o=torch.optim.Adam([p]);torch.exp(p).backward();o.step()
        self.assertTrue(torch.isfinite(p));self.assertTrue(torch.isfinite(o.state[p]['exp_avg_sq']))
    def test_loss_not_exp_of_average(self):
        ct=torch.tensor([1.,1.02],dtype=torch.float64);base=torch.ones(2,dtype=torch.float64)
        loss,z=loss_exponential(ct,base)
        self.assertAlmostEqual(float(loss),(1+np.exp(2))/2)
        self.assertNotAlmostEqual(float(loss),np.exp(1))
    def test_checkpoint_resume(self):
        c=Config(T=6)
        with tempfile.TemporaryDirectory() as tmp:
            a,ma=fit(c,'scaled64',0,tmp+'/a',n=8,epochs=4)
            fit(c,'scaled64',0,tmp+'/b',n=8,epochs=2)
            b,mb=fit(c,'scaled64',0,tmp+'/b',n=8,epochs=4)
            self.assertEqual(ma['status'],'completed');self.assertEqual(mb['status'],'completed')
            for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
if __name__=='__main__':unittest.main(verbosity=2)
