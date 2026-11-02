"""Amendment scope and public-entry routing; no scientific execution."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import prepare
import launch


class AmendmentTests(unittest.TestCase):
    def test_each_scientific_file_change_refused_without_writing_protocol(self):
        for changed in ('protocol.json', 'PROTOCOL.md', 'DATA_PLAN.json'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); old=root/'prior'; new=root/'new'; old.mkdir(); new.mkdir()
                for directory in (old,new):
                    (directory/'protocol.json').write_text('{}')
                    (directory/'PROTOCOL.md').write_text('identical scientific prose')
                    (directory/'DATA_PLAN.json').write_text('{}')
                (new/'AMENDMENT_LINEAGE.json').write_text(json.dumps({'prior_package':'prior'}))
                path=new/changed
                path.write_text('{"changed":true}' if changed.endswith('.json') else 'changed scientific prose')
                before=(new/'protocol.json').read_bytes()
                with patch.object(prepare,'HERE',new), self.assertRaisesRegex(ValueError,'Authoritative scientific file changed'):
                    prepare.freeze()
                self.assertEqual(before,(new/'protocol.json').read_bytes())
                self.assertFalse((new/'PREPARED.json').exists())

    def test_public_validation_is_disabled_in_production_release(self):
        tree=ast.parse(Path(launch.__file__).read_text())
        strings={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertIn('Validated amendment attempt 3 is already sealed; this release cannot repeat validation',strings)

    def test_attempt3_marker_blocks_refreezing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'VALIDATION_ATTEMPT_3.json').write_text('{}')
            with patch.object(prepare,'HERE',root), self.assertRaisesRegex(FileExistsError,'Existing seal or attempt'):
                prepare.freeze()

    def test_actual_public_main_cannot_repeat_validation(self):
        with patch.object(launch.sys,'argv',['launch.py','--validate-runtime']), \
             patch.object(launch,'verify_prepared',return_value=({'p':1},{'m':2})), \
             patch.object(launch,'verify_inputs',return_value={}), \
             patch.object(launch,'execute') as production, \
             patch.object(launch,'module') as load:
            with self.assertRaisesRegex(ValueError,'cannot repeat validation'):
                launch.main()
            load.assert_not_called()
            production.assert_not_called()


if __name__=='__main__': unittest.main()
