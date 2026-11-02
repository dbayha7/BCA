"""Lock-tolerant writes: identical bytes to common's writers, bounded retries, real Windows lock."""
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import common
import lock_tolerant


VALUE = {'b': [1, 2.5, -0.0, 1e-300, 12345678901234567890], 'a': {'z': 'text', 'y': None, 'x': True},
         'unicode': 'Δ load ≥ 0.4', 'nested': [[0.1, 0.2], {'k': 3}]}


class LockTolerantWriteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_bytes_and_descriptor_identical_to_common_writers(self):
        expected = common.atomic_json(self.root/'common.json', VALUE)
        actual = lock_tolerant.lock_tolerant_json(self.root/'tolerant.json', VALUE)
        self.assertEqual((self.root/'common.json').read_bytes(), (self.root/'tolerant.json').read_bytes())
        self.assertEqual(lock_tolerant.serialized(VALUE), (self.root/'common.json').read_bytes())
        self.assertEqual({k: v for k, v in expected.items() if k != 'path'},
                         {k: v for k, v in actual.items() if k != 'path'})
        raw = b'\x00binary\xff'
        self.assertEqual(common.atomic_bytes(self.root/'a.bin', raw)['sha256'],
                         lock_tolerant.lock_tolerant_bytes(self.root/'b.bin', raw)['sha256'])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ['a.bin', 'b.bin', 'common.json', 'tolerant.json'])

    def test_retries_transient_permission_error_with_bounded_backoff(self):
        real, failures, sleeps, stats = os.replace, [3], [], {'lock_errors': 0}
        def flaky(source, target):
            if failures[0]:
                failures[0] -= 1
                raise PermissionError(13, 'Permission denied')
            return real(source, target)
        with patch.object(lock_tolerant.os, 'replace', side_effect=flaky):
            lock_tolerant.lock_tolerant_json(self.root/'progress.json', VALUE, stats=stats, sleep=sleeps.append)
        self.assertEqual(stats['lock_errors'], 3)
        self.assertEqual(sleeps, [.05, .1, .2])
        self.assertEqual((self.root/'progress.json').read_bytes(), lock_tolerant.serialized(VALUE))
        self.assertEqual([p.name for p in self.root.iterdir()], ['progress.json'])

    def test_persistent_lock_raises_after_attempts_and_leaves_destination_untouched(self):
        (self.root/'report.json').write_bytes(b'previous')
        sleeps, stats = [], {'lock_errors': 0}
        with patch.object(lock_tolerant.os, 'replace', side_effect=PermissionError(13, 'Permission denied')):
            with self.assertRaises(PermissionError):
                lock_tolerant.lock_tolerant_json(self.root/'report.json', VALUE, attempts=5, stats=stats,
                                                 sleep=sleeps.append)
        self.assertEqual(stats['lock_errors'], 5)
        self.assertEqual(sleeps, [.05, .1, .2, .4])
        self.assertEqual((self.root/'report.json').read_bytes(), b'previous')
        self.assertEqual(min(.05 * 2 ** 39, 1.), 1.)

    def test_other_os_errors_are_not_retried(self):
        sleeps = []
        with patch.object(lock_tolerant.os, 'replace', side_effect=FileNotFoundError('gone')):
            with self.assertRaises(FileNotFoundError):
                lock_tolerant.lock_tolerant_json(self.root/'x.json', VALUE, sleep=sleeps.append)
        self.assertEqual(sleeps, [])

    @unittest.skipUnless(os.name == 'nt', 'Windows share-mode lock reproduction')
    def test_real_windows_reader_lock_fails_common_writer_but_not_tolerant_writer(self):
        """Reproduces the first attempt's failure: replacing a file another handle holds open."""
        target = self.root/'progress.json'
        common.atomic_json(target, {'status': 'running'})
        reader = open(target, 'rb')  # Python opens without FILE_SHARE_DELETE on Windows.
        try:
            with self.assertRaises(PermissionError):
                common.atomic_json(target, VALUE)
        finally:
            reader.close()
        for leftover in self.root.glob('*.tmp.*'):
            leftover.unlink()
        reader = open(target, 'rb')
        timer = threading.Timer(.3, reader.close); timer.start()
        stats = {'lock_errors': 0}
        try:
            lock_tolerant.lock_tolerant_json(target, VALUE, stats=stats)
        finally:
            timer.join(); reader.close()
        self.assertGreaterEqual(stats['lock_errors'], 1)
        self.assertEqual(target.read_bytes(), lock_tolerant.serialized(VALUE))


if __name__ == '__main__': unittest.main()
