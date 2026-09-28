import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from runtime_bindings import SourceCallableGuard,defaults_hash,digest,token,file_identity,read_source


def fixture_require(ok,message):
    if not ok:raise ValueError(message)


def fixture_packed(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def fixture_sha(value):return hashlib.sha256(fixture_packed(value)).hexdigest()


SOURCE='''TOUCHES.append('loaded')
OFFSET = 2
def helper(x):
    return x + 1
def factory(x=3, *, extra=0):
    return dependency.nested.calc(x) + OFFSET + extra
def use_helper(x):
    return helper(x)
def use_builtin(x):
    return len(x)
def signed_zero(x=-0.0):
    return x
def mutable_default(x=[]):
    return x
def dynamic(x):
    return getattr(dependency, x)
class Factory:
    def __init__(self, value=3):
        self.value = value
    def run(self, x=2):
        return dependency.nested.calc(x) + OFFSET + self.value
    @staticmethod
    def static(x=4):
        return x
    @classmethod
    def classmethod(cls, x=5):
        return x
class Inherited(Factory):
    pass
CONFIG = {'gain': -0.0}
def use_config():
    return CONFIG['gain']
def import_inside():
    import math
    return math.pi
'''


class Fixture:
    def __init__(self,root,source=SOURCE):
        self.path=Path(root)/'synthetic_runtime.py';self.path.write_bytes(source.encode())
        self.module=types.ModuleType('synthetic_runtime');self.module.__file__=str(self.path)
        self.dependency=types.ModuleType('synthetic_dependency');self.nested=types.ModuleType('synthetic_nested')
        self.nested.calc=abs;self.dependency.nested=self.nested
        self.module.__dict__.update(TOUCHES=[],dependency=self.dependency)
        exec(compile(source,str(self.path),'exec',dont_inherit=True),self.module.__dict__)
        self.pin=dict(path=str(self.path),sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),identity=file_identity(self.path))
        self.global_inputs=dict(dependency=self.dependency,OFFSET=2)
        self.attributes={('dependency','nested'):self.nested,('dependency','nested','calc'):abs}
    def args(self,owner_path=(),attribute='factory',defaults=((3,),{'extra':0}),globals_=None,attributes=None):
        return dict(root=self.module,owner_path=owner_path,attribute=attribute,source_pin=copy.deepcopy(self.pin),
                    expected_defaults_sha256=digest(token(defaults)),
                    expected_globals=self.global_inputs if globals_ is None else globals_,
                    expected_attributes=self.attributes if attributes is None else attributes)
    def guard(self,**kwargs):return SourceCallableGuard(**self.args(**kwargs))


class RuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='v2-binding-synthetic-');self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name)
    def test_compile_does_not_execute_source_or_call_factory(self):
        g=self.f.guard();r=g.receipt()
        self.assertEqual(self.f.module.TOUCHES,['loaded']);self.assertTrue(r['binding_current'])
        self.assertFalse(r['execution_accepted']);self.assertFalse(r['dependency_graph_completeness_accepted'])
        self.assertEqual(r['static_attribute_paths'],['dependency.nested','dependency.nested.calc'])
    def test_method_static_class_and_constructor_bindings(self):
        for name,default in [('run',2),('static',4),('classmethod',5),('__init__',3)]:
            g=self.f.guard(owner_path=('Factory',),attribute=name,defaults=((default,),None),
                globals_=self.f.global_inputs if name=='run' else {},attributes=self.f.attributes if name=='run' else {})
            g.assert_unchanged()
        self.assertEqual(self.f.module.TOUCHES,['loaded'])
    def test_same_filename_forged_code_rejected(self):
        original=self.f.module.factory
        bad=compile('def factory(x=3, *, extra=0):\n    return 123\n',str(self.f.path),'exec')
        ns={};exec(bad,ns);original.__code__=ns['factory'].__code__
        with self.assertRaisesRegex(ValueError,'code differs'):self.f.guard()
    def test_live_code_substitution_after_binding(self):
        g=self.f.guard();self.f.module.factory.__code__=self.f.module.use_helper.__code__
        with self.assertRaises(ValueError):g.assert_unchanged()
        self.assertTrue(g.failed)
    def test_external_defaults_required(self):
        with self.assertRaisesRegex(ValueError,'defaults'):self.f.guard(defaults=((99,),{'extra':0}))
    def test_default_mutation_detected(self):
        g=self.f.guard();self.f.module.factory.__defaults__=(99,)
        with self.assertRaises(ValueError):g.assert_unchanged()
    def test_keyword_default_mutation_detected(self):
        g=self.f.guard();self.f.module.factory.__kwdefaults__['extra']=5
        with self.assertRaises(ValueError):g.assert_unchanged()
    def test_mutable_default_contents_detected(self):
        g=self.f.guard(attribute='mutable_default',defaults=(([],),None),globals_={},attributes={})
        self.f.module.mutable_default.__defaults__[0].append(1)
        with self.assertRaises(ValueError):g.assert_unchanged()
    def test_signed_zero_default_preserved(self):
        g=self.f.guard(attribute='signed_zero',defaults=((-0.,),None),globals_={},attributes={})
        self.f.module.signed_zero.__defaults__=(0.,)
        with self.assertRaises(ValueError):g.assert_unchanged()
    def test_wrong_expected_global_before_binding(self):
        with self.assertRaisesRegex(ValueError,'global changed'):
            self.f.guard(globals_=dict(dependency=self.f.dependency,OFFSET=3))
    def test_global_mutation_poisons_even_after_restore(self):
        g=self.f.guard();self.f.module.OFFSET=99
        with self.assertRaises(ValueError):g.assert_unchanged()
        self.f.module.OFFSET=2
        with self.assertRaises(ValueError):g.assert_unchanged()
    def test_global_inventory_omission_and_extra(self):
        for value in ({'OFFSET':2},dict(self.f.global_inputs,UNUSED=1)):
            with self.assertRaisesRegex(ValueError,'global inventory'):self.f.guard(globals_=value)
    def test_static_module_attribute_mutation(self):
        g=self.f.guard();self.f.nested.calc=max
        with self.assertRaisesRegex(ValueError,'attribute changed'):g.assert_unchanged()
    def test_intermediate_module_substitution(self):
        g=self.f.guard();replacement=types.ModuleType('synthetic_nested');replacement.calc=abs
        self.f.dependency.nested=replacement
        with self.assertRaises(ValueError):g.assert_unchanged()
    def test_module_attribute_inventory_omission(self):
        with self.assertRaisesRegex(ValueError,'attribute inventory'):self.f.guard(attributes={('dependency','nested','calc'):abs})
    def test_root_owner_class_replacement(self):
        g=self.f.guard(owner_path=('Factory',),attribute='run',defaults=((2,),None))
        self.f.module.Factory=type('Factory',(),{'run':self.f.module.Factory.run})
        with self.assertRaisesRegex(ValueError,'class replaced'):g.assert_unchanged()
    def test_defining_attribute_replacement(self):
        g=self.f.guard();self.f.module.factory=self.f.module.helper
        with self.assertRaises(ValueError):g.assert_unchanged()
    def test_inherited_owner_not_defining_class(self):
        with self.assertRaisesRegex(ValueError,'defining owner'):
            self.f.guard(owner_path=('Inherited',),attribute='run',defaults=((2,),None))
    def test_referenced_helper_code_substitution(self):
        g=self.f.guard(attribute='use_helper',defaults=(None,None),globals_={'helper':self.f.module.helper},attributes={})
        self.f.module.helper.__code__=self.f.module.signed_zero.__code__
        with self.assertRaisesRegex(ValueError,'global changed'):g.assert_unchanged()
    def test_builtin_shadowing_detected(self):
        g=self.f.guard(attribute='use_builtin',defaults=(None,None),globals_={'len':len},attributes={})
        self.f.module.len=lambda x:123
        with self.assertRaisesRegex(ValueError,'global changed'):g.assert_unchanged()
    def test_source_modified_even_if_code_unchanged(self):
        g=self.f.guard();self.f.path.write_bytes(self.f.path.read_bytes()+b'# changed\n')
        with self.assertRaisesRegex(ValueError,'Source file'):g.assert_unchanged()
    def test_same_bytes_replaced_source_identity(self):
        g=self.f.guard();new=self.f.path.with_name('replacement.py');new.write_bytes(self.f.path.read_bytes());os.replace(new,self.f.path)
        with self.assertRaisesRegex(ValueError,'Source file'):g.assert_unchanged()
    def test_source_hash_mismatch(self):
        args=self.f.args();args['source_pin']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Source bytes'):SourceCallableGuard(**args)
    def test_symlink_and_hardlink_refusal(self):
        link=self.f.path.with_name('alias.py');link.symlink_to(self.f.path)
        with self.assertRaises(ValueError):file_identity(link)
        hard=self.f.path.with_name('hard.py');os.link(self.f.path,hard)
        with self.assertRaises(ValueError):read_source(self.f.pin)
    def test_fork_and_contract_mutation_refusal(self):
        g=self.f.guard()
        with patch('runtime_bindings.os.getpid',return_value=-1):
            with self.assertRaises(ValueError):g.assert_unchanged()
        h=self.f.guard();h.source_pin['sha256']='0'*64
        with self.assertRaises(ValueError):h.assert_unchanged()
    def test_dynamic_namespace_refused(self):
        with self.assertRaisesRegex(ValueError,'Dynamic namespace'):
            self.f.guard(attribute='dynamic',defaults=(None,None),globals_={},attributes={})
    def test_wrong_defining_module_and_closure_refused(self):
        self.f.module.factory.__module__='other'
        with self.assertRaisesRegex(ValueError,'defining module'):self.f.guard()
        def make():
            captured=1
            def factory(x=3,*,extra=0):return captured+x+extra
            return factory
        self.f.module.factory=make()
        with self.assertRaisesRegex(ValueError,'without closure'):self.f.guard()
    def test_changed_function_provenance_metadata(self):
        g=self.f.guard();self.f.module.factory.__qualname__='wrong_name'
        with self.assertRaises(ValueError):g.assert_unchanged()
    def test_mutable_global_contents_and_signed_zero(self):
        g=self.f.guard(attribute='use_config',defaults=(None,None),globals_={'CONFIG':{'gain':-0.}},attributes={})
        self.f.module.CONFIG['gain']=0.
        with self.assertRaises(ValueError):g.assert_unchanged()
    def test_referenced_closure_not_treated_as_bound(self):
        def make():
            state=[1]
            def helper(x):return state[0]+x
            return helper
        self.f.module.helper=make()
        with self.assertRaisesRegex(ValueError,'Referenced closure'):
            self.f.guard(attribute='use_helper',defaults=(None,None),globals_={'helper':self.f.module.helper},attributes={})
    def test_runtime_import_refuses_unbound_local_module(self):
        with self.assertRaisesRegex(ValueError,'Runtime imports'):
            self.f.guard(attribute='import_inside',defaults=(None,None),globals_={},attributes={})
    def test_exact_frozen_archive_code_in_synthetic_namespace(self):
        # No production module import or archive construction. This checks exact
        # class code compiled from its AST against the full source compilation.
        import ast,json,sqlite3,zlib
        from runtime_bindings import requirements,source_code,actual_global,static_member
        path=Path('/mnt/c/Users/David Bayha/Documents/GitHub/BCA/experiments/ood/streaming.py')
        root=Path(__file__).resolve().parents[2]
        pins=json.loads((root/'work/standard_bca_noiw_campaign_v1/monitor_20260928T164055Z/join_external_bindings.json').read_bytes())['source_sha256']
        raw=path.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),pins['experiments/ood/streaming.py'])
        selected=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='ArtifactArchive']
        self.assertEqual(len(selected),1)
        mod=types.ModuleType('synthetic_exact_archive');mod.__file__=str(path)
        mod.__dict__.update(Path=Path,sqlite3=sqlite3,hashlib=hashlib,json=json,zlib=zlib,
                            require=fixture_require,packed=fixture_packed,sha=fixture_sha)
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec',dont_inherit=True),mod.__dict__)
        pin=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),identity=file_identity(path))
        defaults={'__init__':((2000000,),{'read_only':False}),'append':((True,),None),'read':((False,),None)}
        for name,default in defaults.items():
            fn=vars(mod.ArtifactArchive)[name];used,attrs=requirements(fn.__code__)
            expected_globals={n:actual_global(fn,n) for n in used};expected_attrs={}
            for chain in attrs:
                v=expected_globals[chain[0]]
                for part in chain[1:]:v=static_member(v,part)
                expected_attrs[chain]=v
            guard=SourceCallableGuard(root=mod,owner_path=('ArtifactArchive',),attribute=name,source_pin=pin,
                expected_defaults_sha256=digest(token(default)),expected_globals=expected_globals,expected_attributes=expected_attrs)
            self.assertFalse(guard.receipt()['native_runtime_accepted'])
        self.assertFalse((Path(self.tmp.name)/'archive.sqlite').exists())


if __name__=='__main__':unittest.main(verbosity=2)
