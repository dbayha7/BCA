"""Storage and 64-episode collection tests use mock dynamics only."""
from pathlib import Path
from contextlib import closing
import io
import json
import sqlite3
import tempfile
import unittest
import numpy as np

from experiments.ood.streaming import ArtifactArchive, ArchiveDirectory, collect_bank
from experiments.ood.production import check_gate,run


class MockSim:
    def __init__(self):
        self.transitions = 0
    def reset(self, seed):
        self.seed, self.t = seed, 0
        return np.array([0.])
    def capture(self):
        return {'seed': self.seed, 't': self.t}
    def step(self, action):
        self.t += 1; self.transitions += 1
        return dict(observation=np.array([float(self.t)]), terminated=self.seed % 2 == 0 and self.t == 2,
                    truncated=False)


class Policy:
    def actions(self, obs):
        return np.zeros((len(obs),1),np.float32)
    def seal(self):
        return None


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()

    def test_array_bytes_roundtrip_and_exclusive_names(self):
        with ArtifactArchive(self.root/'a.sqlite') as archive:
            directory=ArchiveDirectory(archive)
            with (directory/'transition-00001-input.npz').open('xb') as stream:
                np.savez(stream,a=np.array([0.,-0.,1.],np.float32),state=np.asarray('full-state'))
            with np.load(io.BytesIO(archive.read('transition-00001-input.npz'))) as saved:
                self.assertEqual(saved['a'].tobytes(),np.array([0.,-0.,1.],np.float32).tobytes())
                self.assertEqual(saved['state'].item(),'full-state')
            with self.assertRaises(ValueError):
                with (directory/'transition-00001-input.npz').open('xb') as stream:
                    self.fail('A reused name must fail before writing/callback.')
            self.assertEqual(archive.audit()['records'],1)
        with self.assertRaises(FileExistsError):
            ArtifactArchive(self.root/'a.sqlite')

    def test_invalid_names_and_modes_refused(self):
        with ArtifactArchive(self.root/'a.sqlite') as archive:
            directory=ArchiveDirectory(archive)
            for name in ('../a','/a','a','transition-1.npz'):
                with self.assertRaises(ValueError):
                    directory/name
            with self.assertRaises(ValueError):
                with (directory/'transition-00001.npz').open('wb'):
                    self.fail('Invalid write mode accepted.')

    def test_incomplete_write_stops_and_is_preserved(self):
        with ArtifactArchive(self.root/'a.sqlite') as archive:
            with self.assertRaises(RuntimeError):
                with (ArchiveDirectory(archive)/'transition-00001.npz').open('xb') as stream:
                    stream.write(b'partial'); raise RuntimeError('writer failed')
            self.assertEqual(archive.read('transition-00001.npz',allow_partial=True),b'partial')
            with self.assertRaises(ValueError):
                archive.audit()

    def test_database_corruption_is_detected(self):
        with ArtifactArchive(self.root/'a.sqlite') as archive:
            archive.append('record',b'evidence')
            with closing(sqlite3.connect(self.root/'a.sqlite',isolation_level=None)) as db:
                with self.assertRaises(sqlite3.IntegrityError):db.execute('DELETE FROM artifacts')
            self.assertEqual(archive.audit()['records'],1)

    def test_bounded_payload(self):
        with ArtifactArchive(self.root/'a.sqlite',maximum_record_bytes=3) as archive:
            with self.assertRaises(ValueError):archive.append('too-large',b'1234')
            self.assertEqual(archive.audit()['records'],0)

    def test_closed_archive_read_only_verification(self):
        path=self.root/'a.sqlite'
        with ArtifactArchive(path) as archive:
            archive.append('saved',b'exact');expected=archive.audit()
        with ArtifactArchive(path,read_only=True) as archive:
            self.assertEqual(archive.read('saved'),b'exact')
            self.assertEqual(archive.audit(),expected)
            with self.assertRaises(ValueError):archive.append('new',b'no')

    def test_failed_gate_exit_cannot_authorize_production(self):
        path=self.root/'actual_exit.json';path.write_text(json.dumps({'actual_returncode':1,'timeout':False,'interruption_signal':None}))
        with self.assertRaises(ValueError):check_gate(self.root,self.root,path)

    def test_outcome_dispatch_is_not_implemented_or_inferred(self):
        with self.assertRaises(ValueError):run(self.root,self.root,self.root,'outcomes')

    def test_declared_64_episodes_and_all_missing_captures_kept(self):
        sim=MockSim(); seeds=list(range(101,165)); output=self.root/'states';output.mkdir()
        result=collect_bank(sim,{'host':Policy(),'bca':Policy()},seeds,output,lambda owner,a:sim.step(a))
        self.assertEqual(len(result['rows']),256)
        self.assertEqual(result['captured'],192)
        self.assertEqual(result['missing'],64)
        self.assertEqual(sim.transitions,2*(32*100+32*2))
        self.assertEqual([r['capture_step'] for r in result['rows'][:4]],[0,100,0,100])
        self.assertEqual(result['paired_reset_blocks'],64)
        self.assertEqual(len(list(output.glob('*.json'))),256)

    def test_reduced_or_changed_seed_bank_refused(self):
        for seeds in ([1,2],list(range(63))+[True],list(range(63))+[0]):
            with self.assertRaises(ValueError):
                collect_bank(MockSim(),{'host':Policy(),'bca':Policy()},seeds,self.root,lambda *_:None)


if __name__=='__main__':unittest.main()
