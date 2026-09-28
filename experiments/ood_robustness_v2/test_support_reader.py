import hashlib,json,unittest
import numpy as np
from support_reader import decode_projection


def fixture():
    a=np.zeros((2,3),dtype='<f4')
    enc=dict(__array__=a.tobytes().hex(),dtype=a.dtype.str,shape=list(a.shape))
    return dict(observations=enc.copy(),actions=enc.copy(),validation_observations=enc.copy(),
                validation_actions=enc.copy(),neighbors=2,reference_episodes=[0],validation_episodes=[1,2],validation_distances=[0.,0.])


def decode(v):
    raw=json.dumps(v).encode();return decode_projection(raw,hashlib.sha256(raw).hexdigest())


class Reader(unittest.TestCase):
    def test_roundtrip(self): self.assertEqual(decode(fixture())['actions'].shape,(2,3))
    def test_hash(self):
        with self.assertRaises(ValueError): decode_projection(b'{}','0'*64)
    def test_object_dtype(self):
        v=fixture();v['actions']['dtype']='|O'
        with self.assertRaises(ValueError): decode(v)
    def test_wrong_shape(self):
        v=fixture();v['actions']['shape']=[2,4]
        with self.assertRaises(ValueError): decode(v)
    def test_overlap(self):
        v=fixture();v['reference_episodes']=[1]
        with self.assertRaises(ValueError): decode(v)
    def test_nonfinite(self):
        v=fixture();v['validation_distances'][0]=float('nan')
        with self.assertRaises(ValueError): decode(v)
    def test_duplicate_key(self):
        raw=b'{"x":0,"x":1}'
        with self.assertRaises(ValueError): decode_projection(raw,hashlib.sha256(raw).hexdigest())
    def test_extra_tag(self):
        v=fixture();v['actions']['pickle']='no'
        with self.assertRaises(ValueError): decode(v)


if __name__=='__main__': unittest.main(verbosity=2)
