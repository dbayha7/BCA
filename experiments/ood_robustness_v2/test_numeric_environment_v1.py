"""Pure new-profile refusal cases, no numeric/scientific imports or closed suites."""
import copy,unittest
from numeric_environment_v1 import contract,NumericEnvironmentGuard

class Tests(unittest.TestCase):
    def setUp(self):
        self.policy=contract();self.env=copy.deepcopy(self.policy['environment']);self.kernel=copy.deepcopy(self.env);self.pid=3
    def make(self):return NumericEnvironmentGuard(self.policy,kernel_reader=lambda pid:self.kernel,effective_reader=lambda:self.env,pid_reader=lambda:self.pid)
    def test_exact_phases(self):
        g=self.make()
        for p in self.policy['phases']:self.assertEqual(g.advance(p)['phase'],p)
        g.complete();self.assertTrue(g.completed)
    def test_missing_each_field(self):
        for k in self.policy['environment']:
            with self.subTest(k=k):
                self.env=copy.deepcopy(self.policy['environment']);del self.env[k]
                with self.assertRaises(ValueError):self.make()
    def test_extra_inherited_field(self):
        self.env['PATH']='/bin'
        with self.assertRaises(ValueError):self.make()
    def test_gpu_visibility_changed(self):
        self.env['CUDA_VISIBLE_DEVICES']='0'
        with self.assertRaises(ValueError):self.make()
    def test_effective_change_poison(self):
        g=self.make();self.env['TPU_SKIP_MDS_QUERY']='0'
        with self.assertRaises(ValueError):g.check()
        self.env['TPU_SKIP_MDS_QUERY']='1'
        with self.assertRaises(ValueError):g.check()
    def test_kernel_change_poison(self):
        g=self.make();self.kernel['TF_CPP_MIN_LOG_LEVEL']='0'
        with self.assertRaises(ValueError):g.check()
        self.kernel['TF_CPP_MIN_LOG_LEVEL']='1'
        with self.assertRaises(ValueError):g.check()
    def test_policy_mutation(self):
        g=self.make();g.policy['environment']['LC_CTYPE']='C'
        with self.assertRaises(ValueError):g.check()
    def test_policy_does_not_expand(self):
        self.policy['environment']['LD_LIBRARY_PATH']='/tmp'
        with self.assertRaises(ValueError):self.make()
    def test_native_acceptance_refused(self):
        self.policy['native_environment_accepted']=True
        with self.assertRaises(ValueError):self.make()
    def test_process_change(self):
        g=self.make();self.pid=4
        with self.assertRaises(ValueError):g.check()
    def test_skipped_phase(self):
        g=self.make()
        with self.assertRaises(ValueError):g.advance('jax')
        with self.assertRaises(ValueError):g.advance('python')
    def test_repeat_phase(self):
        g=self.make();g.advance('python')
        with self.assertRaises(ValueError):g.advance('python')
    def test_complete_early(self):
        g=self.make()
        with self.assertRaises(ValueError):g.complete()
    def test_complete_twice(self):
        g=self.make()
        for p in self.policy['phases']:g.advance(p)
        g.complete()
        with self.assertRaises(ValueError):g.complete()
    def test_input_policy_copy(self):
        g=self.make();self.policy['environment']['JAX_PLATFORMS']='cuda';g.check()
    def test_failed_reader_poison(self):
        g=self.make();g.kernel_reader=lambda pid:None
        with self.assertRaises(ValueError):g.check()
        g.kernel_reader=lambda pid:self.kernel
        with self.assertRaises(ValueError):g.check()

if __name__=='__main__':unittest.main()
