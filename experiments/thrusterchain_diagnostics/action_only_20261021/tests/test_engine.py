"""Bounded genuine-checkpoint fixtures; no fitting or experiment sampling."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engine
engine.configure_runtime()
import numpy as np

TASK=Path(__file__).resolve().parents[2]
BUNDLE=TASK/"results_20261021/release/ThrusterChain_results_20261021"
REPO=Path(r"C:\Users\David\dev\BCA")
RUN=REPO/"experiments/thrusterchain_diagnostics/training_runs/core_20261020T2250Z"


class GridTests(unittest.TestCase):
    def test_enumeration_projection_and_ties(self):
        atoms=np.array([-1.,float(np.float32(-.8)),0.,float(np.float32(.8)),1.])
        grid,indices=engine.action_grid(atoms)
        np.testing.assert_array_equal(engine.project_actions(grid,atoms),np.arange(625))
        np.testing.assert_array_equal(indices[0],[0,0,0,0])
        np.testing.assert_array_equal(indices[-1],[4,4,4,4])
        midpoint=np.full((1,4),atoms[1]/2)
        self.assertEqual(int(engine.project_actions(midpoint,atoms)[0]),156) # all coordinate index1

    def test_outcome_blind_selection_and_context_times(self):
        episodes=np.repeat(np.arange(80),100)
        time=np.tile(np.arange(100),80)
        rows=dict(episode_id=episodes,time=time,context=episodes//20,row_id=np.arange(8000))
        for k in ("observation","state","action","next_observation","action_atom_index"):
            rows[k]=np.zeros((8000,15 if "observation" in k else 13 if k=="state" else 4))
        for k in ("regime","reward","action_probability"):rows[k]=np.zeros(8000)
        a=engine.select_states(rows,2026102101)
        rows["reward"]=np.arange(8000)*-1e6
        rows["state"][:]=9000
        b=engine.select_states(rows,2026102101)
        np.testing.assert_array_equal(a["row_id"],b["row_id"])
        self.assertEqual(len(np.unique(a["episode_id"])),64)
        for c in range(4):
            self.assertEqual(np.bincount(a["time"][a["context"]==c],minlength=100)[[0,50,99]].tolist(),[6,5,5])


class FrozenCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=engine.FrozenScoreEngine.load(bundle_root=BUNDLE,canonical_run_dir=RUN,
                                           state_seed=2026102101,repository_root=REPO)

    def test_identity_support_and_mixture(self):
        e=self.e
        self.assertEqual(e.metadata["checkpoint"]["sha256"],"4088278230fc413c509e27e189daa342468cfc4fb6d2479e3e8c29e149dac3d8")
        self.assertEqual(e.reference_update,95000)
        np.testing.assert_allclose(e.beta.sum(1),1,rtol=0,atol=2e-14)
        self.assertAlmostEqual(float(e.beta.min()),.04**4)
        np.testing.assert_array_equal(e.target_probabilities(0),e.beta)
        target=e.target_probabilities(1)
        self.assertTrue(np.all((target==0)|(target==1)))
        np.testing.assert_array_equal(target.argmax(1),e.actor_action_index)

    def test_selected_logged_transition_and_beta_parity(self):
        e=self.e
        a=engine.project_actions(e.selected["action"],e.atoms)
        t=e.transitions(np.arange(64),a)
        raw_next=(e.selected["next_observation"].astype(np.float32)-e.result.mean)/e.result.std
        np.testing.assert_array_equal(t["next_obs"],raw_next)
        np.testing.assert_array_equal(t["reward"],e.selected["reward"].astype(np.float32))
        np.testing.assert_allclose(e.beta[np.arange(64),a],e.selected["action_probability"],rtol=0,atol=1e-15)

    def test_native_target_score_and_frozen_scale_parity(self):
        import jax
        import jax.numpy as jnp
        from algorithms import td3_bc_bca as P
        from calibration.reference import positive_scale
        from runtime.networks import Transition
        e=self.e
        z=np.array([0,1,2,16,17,18],np.int64)
        a=np.array([0,156,312,468,624,312],np.int64)
        t=e.transitions(z,a)
        key=jax.random.PRNGKey(20261021)
        noise=np.asarray(jax.random.normal(key,(len(z),4)))
        b=Transition(**{k:jnp.asarray(v) for k,v in t.items()})
        native=np.asarray(jax.jit(lambda batch:P.native_target(e.args,e.result.models,e.result.state.native,batch,key))(b))
        result=e.score(z,a,noise)
        np.testing.assert_allclose(result.target,native,rtol=1e-6,atol=1e-6)
        q=np.asarray(e.result.models[1].apply(e.result.state.native.critic.params,b.obs,b.action)).min(-1)
        eta=e.result.models[2].apply(e.result.state.posterior.cal_params,b.obs,b.action)
        scale=np.asarray(positive_scale(eta,e.result.state.posterior.residual_scale))
        residual=np.asarray(native-q,dtype=np.float32)
        expected=np.abs(residual.astype(np.float64))/scale.astype(np.float64)
        np.testing.assert_allclose(result.q,q,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(result.scale,scale,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(result.scores,expected,rtol=2e-5,atol=2e-5)
        np.testing.assert_array_equal(result.target[z%16==2],t["reward"][z%16==2])

    def test_no_state_mutation_and_label_noise_only(self):
        e=self.e
        before=e.T.tree_sha256(e.result.carry)
        z=np.array([0,1,2],np.int64);a=np.array([312,312,312],np.int64)
        r1=e.score(z,a,np.zeros((3,4),np.float32))
        r2=e.score(z,a,np.ones((3,4),np.float32))
        np.testing.assert_array_equal(r1.scale,r2.scale)
        np.testing.assert_array_equal(r1.q,r2.q)
        self.assertNotEqual(float(r1.target[0]),float(r2.target[0]))
        self.assertEqual(float(r1.target[2]),float(r2.target[2]))
        self.assertEqual(before,e.T.tree_sha256(e.result.carry))
        with self.assertRaises(ValueError):e.score(z,a,np.zeros((3,3)))


if __name__=="__main__":
    unittest.main(verbosity=2)
