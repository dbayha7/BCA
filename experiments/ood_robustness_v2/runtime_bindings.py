"""Live Python callable/source binding without executing the pinned source.

Expected object references and graph completeness require independent acceptance.
Native/JIT internals and imported dependency inventories are not certified here.
There is no scientific import, dispatch, ledger construction or default real path.
"""
import builtins
import dis
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import types


def require(ok,message):
    if not ok: raise ValueError(message)


def digest(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def token(value,depth=0):
    require(depth<=30,'Literal depth limit.')
    t=type(value)
    if value is Ellipsis:return ['ellipsis']
    if value is None:return ['none']
    if t is str:require(len(value)<=2_000_000,'Literal string bound.')
    if t in (str,bool,int):return [t.__name__,value]
    if t is float:return ['float',value.hex()]
    if t is complex:return ['complex',value.real.hex(),value.imag.hex()]
    if t is bytes:
        require(len(value)<=2_000_000,'Literal byte bound.');return ['bytes',value.hex()]
    if t in (tuple,list):
        require(len(value)<=10000,'Literal item bound.');return [t.__name__,[token(v,depth+1) for v in value]]
    if t is frozenset:return ['frozenset',sorted([token(v,depth+1) for v in value],key=lambda x:json.dumps(x,sort_keys=True))]
    if t is dict:
        require(len(value)<=1000 and all(type(k) is str for k in value),'Bounded string-key literal mapping required.')
        return ['dict',{k:token(v,depth+1) for k,v in value.items()}]
    if t is types.CodeType:return ['code',code_tree(value,depth+1)]
    raise ValueError('Unsupported mutable/native default or literal; separate binding required.')


def code_tree(code,depth=0):
    require(type(code) is types.CodeType and depth<=30,'Bounded Python code required.')
    fields=('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize','co_flags',
            'co_code','co_names','co_varnames','co_freevars','co_cellvars','co_name','co_filename','co_firstlineno',
            'co_lnotab','co_linetable','co_exceptiontable','co_qualname')
    out={n:token(getattr(code,n),depth+1) for n in fields if hasattr(code,n)}
    out['co_consts']=[token(v,depth+1) for v in code.co_consts];return out


def defaults_hash(fn): return digest(token((fn.__defaults__,fn.__kwdefaults__)))


def file_identity(path):
    p=Path(path)
    require(p.is_absolute() and '..' not in p.parts and str(p)==str(path),'Canonical absolute source path required.')
    require(p.parent.resolve(strict=True)==p.parent and p.resolve(strict=True)==p,'Redirected source path.')
    s=p.lstat();require(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'Unaliased regular source required.')
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def read_source(pin):
    require(type(pin) is dict and set(pin)=={'path','sha256','identity'},'Exact external source pin required.')
    require(type(pin['sha256']) is str and len(pin['sha256'])==64 and all(c in '0123456789abcdef' for c in pin['sha256']),'External source hash required.')
    require(type(pin['identity']) is list and len(pin['identity'])==5 and all(type(n) is int for n in pin['identity']),'External stat identity required.')
    p=Path(pin['path']);before=file_identity(pin['path'])
    require(before==pin['identity'] and 0<before[2]<=2_000_000,'Source identity/size mismatch.')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        s=os.fstat(f.fileno());require([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==before,'Source descriptor replaced.')
        raw=f.read(2_000_001)
    require(file_identity(p)==before and hashlib.sha256(raw).hexdigest()==pin['sha256'],'Source bytes changed.')
    return raw


def source_code(module_code,qualname):
    code=module_code
    for name in qualname.split('.'):
        require(name!='<locals>','Closure/nested function requires a separate binding.')
        matches=[v for v in code.co_consts if type(v) is types.CodeType and v.co_name==name]
        require(len(matches)==1,'Defining source code is absent or ambiguous.');code=matches[0]
    return code


def requirements(code):
    globals_used=set();attributes=set()
    todo=[code]
    while todo:
        c=todo.pop();instructions=list(dis.get_instructions(c))
        for i,op in enumerate(instructions):
            require(op.opname not in ('IMPORT_NAME','IMPORT_FROM','IMPORT_STAR'),
                    'Runtime imports require separately bound import/runtime integration.')
            if op.opname=='LOAD_GLOBAL':
                name=op.argval;globals_used.add(name);path=(name,)
                for following in instructions[i+1:]:
                    if following.opname not in ('LOAD_ATTR','LOAD_METHOD'):break
                    path=(*path,following.argval);attributes.add(path)
        todo.extend(v for v in c.co_consts if type(v) is types.CodeType)
    require(not globals_used&{'eval','exec','globals','locals','getattr','setattr','delattr','__import__','compile'},
            'Dynamic namespace access requires independent specialized binding.')
    return globals_used,attributes


def static_member(owner,name):
    require(type(name) is str and not name.startswith('__'),'Unsafe binding attribute.')
    value=inspect.getattr_static(owner,name)
    require(not isinstance(value,property),'Property evaluation is not a static binding.')
    return value


def reference_token(value):
    if type(value) is types.FunctionType:
        require(value.__closure__ is None,'Referenced closure requires separate binding.')
        return ['function',id(value),value.__module__,value.__qualname__,value.__name__,
                digest(code_tree(value.__code__)),defaults_hash(value)]
    if isinstance(value,(staticmethod,classmethod)):return [type(value).__name__,reference_token(value.__func__)]
    if isinstance(value,types.ModuleType):return ['module',id(value),vars(value).get('__name__'),vars(value).get('__file__')]
    if isinstance(value,type):return ['type',id(value)]
    if inspect.isbuiltin(value) or isinstance(value,(types.MethodDescriptorType,types.WrapperDescriptorType)):
        return ['native_callable',id(value)]
    return ['literal',token(value)]


def actual_global(fn,name):
    if name in fn.__globals__:return fn.__globals__[name]
    namespace=fn.__builtins__
    if type(namespace) is dict:return namespace[name]
    require(isinstance(namespace,types.ModuleType),'Unexpected builtins namespace.')
    return vars(namespace)[name]


class SourceCallableGuard:
    """One exact live Python function/method and its declared direct bindings.

    Never calls the function. A function loaded from the right filename but with
    different code/defaults/globals fails. Source/global graph provenance remains
    an externally supplied prerequisite, not something these pins self-certify.
    """
    def __init__(self,*,root,owner_path,attribute,source_pin,expected_defaults_sha256,
                 expected_globals,expected_attributes):
        self.failed=False;self.pid=os.getpid();self.root=root
        require(isinstance(root,types.ModuleType),'Explicit defining module root required.')
        require(type(owner_path) is tuple and all(type(n) is str for n in owner_path),'Static owner path required.')
        self.owner_path=owner_path;self.attribute=attribute
        self.source_pin=json.loads(json.dumps(source_pin));self._pin_digest=digest(self.source_pin)
        raw=read_source(self.source_pin);owner=self._owner()
        require(attribute in vars(owner),'Exact defining owner required; inherited methods refused.')
        member=inspect.getattr_static(owner,attribute)
        fn=member.__func__ if isinstance(member,(staticmethod,classmethod)) else member
        require(type(fn) is types.FunctionType and fn.__closure__ is None,'Plain Python function without closure required.')
        require(fn.__module__==vars(root)['__name__'],'Wrong defining module.')
        expected_name='.'.join((*owner_path,attribute))
        require(fn.__qualname__==expected_name,'Wrong defining qualified name.')
        require(fn.__code__.co_filename==self.source_pin['path'],'Source location alias refused.')
        compiled=compile(raw,self.source_pin['path'],'exec',dont_inherit=True,optimize=sys.flags.optimize)
        expected_code=source_code(compiled,expected_name)
        require(code_tree(fn.__code__)==code_tree(expected_code),'Live callable code differs from pinned source.')
        require(defaults_hash(fn)==expected_defaults_sha256,'External defaults binding mismatch.')
        g,a=requirements(expected_code)
        require(type(expected_globals) is dict and set(expected_globals)==g,'Exact referenced-global inventory required.')
        require(type(expected_attributes) is dict and set(expected_attributes)==a,'Exact static attribute inventory required.')
        self.fn=fn;self.owner=owner;self.member_token=reference_token(member)
        self.function_token=reference_token(fn);self._root_token=reference_token(root)
        self.globals={n:reference_token(v) for n,v in expected_globals.items()}
        self.attributes={p:reference_token(v) for p,v in expected_attributes.items()}
        self.expected_defaults_sha256=expected_defaults_sha256
        self._contract_digest=digest([self.globals,sorted((list(p),v) for p,v in self.attributes.items()),self.expected_defaults_sha256])
        self.assert_unchanged()

    def _owner(self):
        owner=self.root
        for name in self.owner_path:
            owner=static_member(owner,name)
            require(isinstance(owner,type),'Only static defining class owners supported.')
        return owner

    def assert_unchanged(self):
        try:
            require(not self.failed and self.pid==os.getpid(),'Failed/forked runtime binding.')
            require(digest(self.source_pin)==self._pin_digest and file_identity(self.source_pin['path'])==self.source_pin['identity'],'Source file/declaration changed.')
            require(reference_token(self.root)==self._root_token and self._owner() is self.owner,'Defining module/class replaced.')
            require(reference_token(inspect.getattr_static(self.owner,self.attribute))==self.member_token
                    and reference_token(self.fn)==self.function_token,'Live callable/defaults changed.')
            require(digest([self.globals,sorted((list(p),v) for p,v in self.attributes.items()),self.expected_defaults_sha256])==self._contract_digest,'Binding contract mutated.')
            for n,t in self.globals.items():require(reference_token(actual_global(self.fn,n))==t,'Runtime global changed: '+n)
            for path,t in self.attributes.items():
                value=actual_global(self.fn,path[0])
                for name in path[1:]:value=static_member(value,name)
                require(reference_token(value)==t,'Runtime module attribute changed: '+'.'.join(path))
        except BaseException:
            self.failed=True;raise

    def receipt(self):
        self.assert_unchanged()
        return dict(schema='ood-v2-runtime-callable-binding-v1',source_pin=json.loads(json.dumps(self.source_pin)),
            module=vars(self.root)['__name__'],qualname=self.fn.__qualname__,code_sha256=digest(code_tree(self.fn.__code__)),
            defaults_sha256=self.expected_defaults_sha256,referenced_globals=sorted(self.globals),
            static_attribute_paths=['.'.join(p) for p in sorted(self.attributes)],
            interpreter=dict(version=list(sys.version_info[:3]),cache_tag=sys.implementation.cache_tag,optimize=sys.flags.optimize),
            binding_current=True,dependency_graph_completeness_accepted=False,native_runtime_accepted=False,
            execution_accepted=False,scientific_acceptance=False)
