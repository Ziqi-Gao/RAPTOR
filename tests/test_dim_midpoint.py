"""Independent mathematical examples required by the midpoint accuracy patch."""
import unittest
import numpy as np
from raptor.probes.dim_midpoint import fit_midpoint,from_means,predict,scores


class MidpointMathTests(unittest.TestCase):
    def test_required_unequal_class_hand_calculation_and_boundary(self):
        x=np.array([[4.,1.],[5.,1.],[6.,1.],[0.,1.],[2.,1.]])
        y=np.array([1,1,1,0,0]);fit=fit_midpoint(x,y,np.arange(5))
        np.testing.assert_array_equal(fit['mean_positive'],[5.,1.]);np.testing.assert_array_equal(fit['mean_negative'],[1.,1.])
        np.testing.assert_array_equal(fit['mean_difference'],[4.,0.]);np.testing.assert_array_equal(fit['midpoint'],[3.,1.]);self.assertEqual(fit['b'],-12.)
        self.assertFalse(np.array_equal(fit['midpoint'],x.mean(axis=0)))
        h=np.array([[2.5,9.],[3.2,-4.],[3.,5.]])
        np.testing.assert_array_equal(predict(h,np.arange(3),fit),[0,1,1])
        u=fit['W']/np.linalg.norm(fit['W']);bu=fit['b']/np.linalg.norm(fit['W'])
        np.testing.assert_array_equal((h@u+bu>=0).astype(int),predict(h,np.arange(3),fit))
        self.assertNotEqual((h@u+fit['b']>=0).astype(int).tolist(),predict(h,np.arange(3),fit).tolist())
        self.assertNotEqual((h@fit['W']>=0).astype(int).tolist(),predict(h,np.arange(3),fit).tolist())

    def test_independent_squared_distance_reference_and_translation(self):
        rng=np.random.RandomState(9);x=rng.normal(size=(100,4));y=np.r_[np.ones(63,dtype=int),np.zeros(37,dtype=int)];x[y==1]+=[1.,2.,-.3,.4]
        train=np.r_[np.arange(45),np.arange(63,90)];fit=fit_midpoint(x,y,train);h=rng.normal(size=(20,4));reference_x=h.astype(np.longdouble)
        pos=x[train[y[train]==1]].astype(np.longdouble).mean(axis=0);neg=x[train[y[train]==0]].astype(np.longdouble).mean(axis=0)
        distance=((reference_x-neg)**2).sum(axis=1)-((reference_x-pos)**2).sum(axis=1)
        np.testing.assert_allclose(2*scores(h,np.arange(20),fit),np.asarray(distance,dtype=float),rtol=2e-14,atol=2e-14)
        np.testing.assert_array_equal(predict(h,np.arange(20),fit),(distance>=0).astype(int))
        shift=np.array([20.,-8.,11.,4.]);translated=fit_midpoint(x+shift,y,train)
        np.testing.assert_array_equal(predict(h+shift,np.arange(20),translated),predict(h,np.arange(20),fit))

    def test_two_dimension_unequal_scale_counterexample(self):
        center=np.array([3.,-5.]);scale=np.array([10.,1.]);pos_z=np.array([.5,.5]);neg_z=-pos_z
        dz=pos_z-neg_z;mz=(pos_z+neg_z)/2
        fit=from_means(center+scale*neg_z,center+scale*pos_z,{'positive':3,'negative':2,'total':5})
        z=np.array([[.2,-1.],[-.2,1.],[0.,0.]])
        raw=center+z*scale
        np.testing.assert_array_equal(fit['mean_difference'],scale*dz)
        np.testing.assert_array_equal(fit['midpoint'],center+scale*mz)
        np.testing.assert_allclose(scores(raw,np.arange(3),fit),(z-mz)@(scale**2*dz),rtol=0,atol=1e-14)
        np.testing.assert_array_equal(predict(raw,np.arange(3),fit),[1,0,1])
        self.assertEqual(float((z-mz)[0]@dz),-.8)
        self.assertEqual(int(predict(raw,np.arange(3),fit)[0]),1)
        with self.assertRaises(ValueError):from_means(neg_z,pos_z,{'positive':3,'negative':2,'total':5},coordinate_space='standardized')

    def test_zero_direction_is_explicit_all_positive_without_normalization(self):
        fit=fit_midpoint(np.array([[2.,3.],[2.,3.]]),np.array([0,1]),np.array([0,1]))
        self.assertTrue(fit['degenerate']);self.assertEqual(fit['b'],0.)
        h=np.array([[999.,-3.],[-7.,2.]])
        np.testing.assert_array_equal(scores(h,np.arange(2),fit),[0.,0.]);np.testing.assert_array_equal(predict(h,np.arange(2),fit),[1,1])

    def test_missing_class_empty_dimension_and_nonfinite_fail(self):
        x=np.array([[1.,2.],[3.,4.]]);y=np.array([0,1])
        with self.assertRaises(ValueError):fit_midpoint(x,y,np.array([],dtype=int))
        with self.assertRaises(ValueError):fit_midpoint(x,y,np.array([0]))
        with self.assertRaises(ValueError):fit_midpoint(x,np.array([1]),np.array([0,1]))
        bad=x.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):fit_midpoint(bad,y,np.array([0,1]))
        fit=fit_midpoint(x,y,np.array([0,1]))
        with self.assertRaises(ValueError):scores(np.zeros((2,3)),np.arange(2),fit)
        with self.assertRaises(ValueError):scores(bad,np.arange(2),fit)

    def test_final_means_use_true_class_counts_and_holdout_not_fit(self):
        train=np.array([[0.,0.],[2.,0.],[4.,0.],[10.,0.]])
        y=np.array([0,0,0,1]);val=np.array([[20.,0.],[22.,0.],[24.,0.],[8.,0.]])
        yv=np.array([1,1,1,0]);x=np.r_[train,val,[[999.,999.]]];labels=np.r_[y,yv,[1]]
        a=fit_midpoint(x,labels,np.arange(4));b=fit_midpoint(x,labels,np.arange(8))
        np.testing.assert_array_equal(b['mean_positive'],np.array([19.,0.]))
        np.testing.assert_array_equal(b['mean_negative'],np.array([3.5,0.]))
        valfit=fit_midpoint(val,yv,np.arange(4))
        self.assertNotEqual(float(b['mean_positive'][0]),float((a['mean_positive'][0]+valfit['mean_positive'][0])/2))
        poisoned=x.copy();poisoned[8]=np.nan
        again=fit_midpoint(poisoned,labels,np.arange(8));np.testing.assert_array_equal(b['midpoint'],again['midpoint'])


if __name__=='__main__':unittest.main()
