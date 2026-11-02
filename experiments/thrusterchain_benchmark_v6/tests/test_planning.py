import unittest
from unittest.mock import patch
import numpy as np
from thrusterchain.environment import Physics, dynamics, matrices, thrust
from thrusterchain.planning import BoundedPlanner, identify_from_training

class PlannerTests(unittest.TestCase):
    def test_observation_only_identification_no_oracle_access(self):
        rng=np.random.default_rng(31)
        A=np.eye(13)*.8; B=rng.normal(scale=.1,size=(13,4))
        x=rng.normal(size=(80,13)); u=rng.uniform(-1,.4,size=(80,4))
        observations=np.zeros((80,2,14),dtype=np.float32)
        observations[:,0,:13]=x; observations[:,1,:13]=x@A.T+u@B.T
        class Logged:
            metadata={"physics":{"stall":False}}; digest="fixture"
            def split(self,name):
                if name!="train": raise AssertionError("non-train access")
                return {"observations":observations,"actions":u[:,None],"group_ids":np.arange(80)}
        with patch("thrusterchain.planning.matrices",side_effect=AssertionError("oracle access")):
            learned=identify_from_training(Logged())
            planner=BoundedPlanner(learned.A,learned.B,Physics(stall=False),2,
                                   provenance="train_only_identified_model")
            planner.plan(np.zeros(13),2)
        np.testing.assert_allclose(learned.A,A,atol=1e-6)
        np.testing.assert_allclose(learned.B,B,atol=1e-6)
        # Omitted action region can be predicted under this correct global model class.
        held=rng.normal(size=(20,13)); omitted=rng.uniform(.6,1,(20,4))
        np.testing.assert_allclose(held@learned.A.T+omitted@learned.B.T,
                                   held@A.T+omitted@B.T,atol=2e-6)
        with self.assertRaises(ValueError): learned.A.setflags(write=True)
        with self.assertRaises(AttributeError): planner.horizon=7

    def test_nonfinite_condensed_objective_refused(self):
        with np.errstate(over="ignore",invalid="ignore"):
            with self.assertRaises(RuntimeError):
                BoundedPlanner(np.eye(13)*1e308,np.ones((13,4)),planning_horizon=5).plan(np.ones(13),5)

    def test_misspecification_is_visible_in_fit_residual(self):
        rng=np.random.default_rng(913)
        states=rng.normal(size=(120,13)); actions=rng.uniform(-1,1,(120,4))
        clean=.8*states
        class Fixture:
            metadata={"physics":{"stall":False}}; digest="misspecification_fixture"
            def __init__(self,target):
                self.obs=np.zeros((120,2,14),np.float32)
                self.obs[:,0,:13]=states;self.obs[:,1,:13]=target
            def split(self,name):
                if name!="train": raise AssertionError("unexpected split")
                return dict(observations=self.obs,actions=actions[:,None],group_ids=np.arange(120))
        base=identify_from_training(Fixture(clean))
        distorted=clean.copy(); distorted[:,0]+=.2*states[:,0]**2
        misspecified=identify_from_training(Fixture(distorted))
        self.assertGreater(misspecified.receipt["training_rmse"],.02)
        self.assertLess(base.receipt["training_rmse"],1e-6)

    def test_last_action_zero(self):
        p = BoundedPlanner(planning_horizon=100)
        np.testing.assert_allclose(p.plan(np.arange(13)*.01,1), np.zeros(4), atol=1e-8)
        self.assertTrue(p.last_certificate["full_remaining_horizon"])

    def test_inactive_bounds_match_independent_riccati(self):
        physics=Physics(stall=False)
        A,B=matrices(physics); Q=np.diag([1.]*10+[0.]*3); R=np.eye(4)*.3
        P=np.zeros((13,13)); first_gain=None
        for t in reversed(range(7)):
            gain=np.linalg.solve(R+physics.gamma*B.T@P@B,physics.gamma*B.T@P@A)
            P=Q+physics.gamma*A.T@P@A-physics.gamma*A.T@P@B@gain
            if t==0: first_gain=gain
        x=np.arange(13)*.0001
        expected=-first_gain@x
        self.assertTrue(np.all(np.abs(expected)<.1))
        actual=BoundedPlanner(physics=physics,planning_horizon=7).plan(x,7)
        np.testing.assert_allclose(actual,expected,atol=2e-6)

    def test_horizon_two_independent_exact_solution(self):
        physics = Physics(stall=False)
        A,B=matrices(physics)
        x=np.zeros(13); x[:10]=np.arange(10)*2; x[10]=1.0
        p=BoundedPlanner(physics=physics,planning_horizon=2)
        u=p.plan(x,2)
        from scipy.optimize import lsq_linear
        M=np.vstack((np.sqrt(physics.throttle_cost)*np.eye(4),
                     np.sqrt(physics.gamma)*B[:10]))
        target=np.r_[np.zeros(4),-np.sqrt(physics.gamma)*(A@x)[:10]]
        reference=lsq_linear(M,target,bounds=(-1,1),tol=1e-13).x
        np.testing.assert_allclose(u,reference,atol=2e-6)
        self.assertLessEqual(p.last_certificate["projected_gradient_residual"],1e-5)

    def test_stall_commands_dominated(self):
        p=Physics()
        rng=np.random.default_rng(4)
        x=rng.normal(size=(500,13)); a=rng.uniform(-1,1,(500,4))
        effective=thrust(a,p)
        self.assertTrue(np.all(np.abs(effective)<=p.knee+1e-14))
        nx,r=dynamics(x,a,p); nx2,r2=dynamics(x,effective,p)
        np.testing.assert_array_equal(nx,nx2)
        self.assertTrue(np.all(r2>=r-1e-12))

    def test_full_remaining_horizon_and_short_mpc_labels(self):
        p=BoundedPlanner(planning_horizon=3)
        p.plan(np.ones(13)*.1,5)
        self.assertFalse(p.last_certificate["full_remaining_horizon"])
        self.assertFalse(p.last_certificate["float32_lattice_optimality"])

    def test_invalid_model_and_rank_refusal(self):
        with self.assertRaises(ValueError): BoundedPlanner(A=np.zeros((3,3)))
        with self.assertRaises(ValueError): BoundedPlanner(physics=Physics(stall_slope=20))
        class BadDataset:
            metadata={"physics":Physics().as_dict()}; digest="test"
            def split(self, name):
                return dict(observations=np.zeros((1,101,14)),actions=np.zeros((1,100,4)),group_ids=np.array([0]))
        with self.assertRaisesRegex(ValueError,"rank"): identify_from_training(BadDataset())

if __name__=="__main__": unittest.main()

