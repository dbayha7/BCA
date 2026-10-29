"""Authenticated receipt integration tests; no model/data/simulator execution."""
import copy
from pathlib import Path
import tempfile
import unittest
import launch


def write(root,relative,value):
    path=root/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    launch.atomic_json(path,value)
    return launch.sha(path)


def fixtures(root):
    protocol=launch.read(launch.HERE/'protocol.json')
    protocol_sha=write(root,'protocol.json',protocol)
    for index in range(3):
        for condition in protocol['conditions']:
            cell=f'seed{index}_{condition}'
            folder=Path('run')/cell/'warmup'
            actor={'cell':cell,'normalization':{'mean':[0.]*14,'std':[1.]*14,
                    'rows':60924,'fixture_digest':'same_observations','source':'common_retained_intersection'}}
            data={'condition':condition,'common_normalization':{'rows':60924,
                    'source_row_ids_sha256':'same_ids','observation_bytes_sha256':'same_observations'}}
            report={'status':'completed_central_quartile_warmup','execution_counts_known':True,
                'cell':cell,'condition':condition,'seed_index':index,'protocol_sha256':protocol_sha,
                'initial_parameter_hashes':{'actor':f'actor{index}','critic':f'critic{index}','auxiliary_scale':f'aux{index}'},
                'initial_optimizer_steps':{'actor':0,'critic':0,'auxiliary':0},
                'initial_optimizer_hashes':{'actor':'zero_actor_moments','critic':'zero_critic_moments','auxiliary':'zero_aux_moments'},
                'actor_manifest_sha256':write(root,folder/'actor_manifest.json',actor),
                'data_receipt_sha256':write(root,folder/'data_receipt.json',data)}
            seal=write(root,folder/'report.json',report)
            write(root,folder/'SUPERVISOR_RESULT.json',{'completed':True,'report_sha256':seal})
    return protocol


def mutate_and_reseal(root,cell,kind,change):
    folder=root/'run'/cell/'warmup'
    artifact=launch.read(folder/(kind+'.json'))
    change(artifact)
    launch.atomic_json(folder/(kind+'.json'),artifact)
    report=launch.read(folder/'report.json')
    if kind=='report': report=artifact
    else: report[kind+'_sha256']=launch.sha(folder/(kind+'.json'))
    launch.atomic_json(folder/'report.json',report)
    launch.atomic_json(folder/'SUPERVISOR_RESULT.json',{'completed':True,'report_sha256':launch.sha(folder/'report.json')})


class PairedTrainingIntegrationTests(unittest.TestCase):
    def test_valid_complete_cell_matrix(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='pair_fixture_') as directory:
            root=Path(directory);protocol=fixtures(root)
            result=launch.verify_warmup_pairs(protocol,root)
            self.assertTrue(result['common_normalization_equal'])
            self.assertTrue(result['initial_optimizer_states_equal'])

    def test_hash_valid_normalization_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='pair_fixture_') as directory:
            root=Path(directory);protocol=fixtures(root)
            mutate_and_reseal(root,'seed0_hole','actor_manifest',lambda x:x['normalization']['mean'].__setitem__(0,.5))
            with self.assertRaisesRegex(ValueError,'normalization differs'):
                launch.verify_warmup_pairs(protocol,root)

    def test_same_statistics_different_rows_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='pair_fixture_') as directory:
            root=Path(directory);protocol=fixtures(root)
            mutate_and_reseal(root,'seed1_random','data_receipt',lambda x:x['common_normalization'].__setitem__('source_row_ids_sha256','different_rows'))
            with self.assertRaisesRegex(ValueError,'provenance differs'):
                launch.verify_warmup_pairs(protocol,root)

    def test_zero_step_with_nonzero_or_different_optimizer_state_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='pair_fixture_') as directory:
            root=Path(directory);protocol=fixtures(root)
            mutate_and_reseal(root,'seed2_full','report',lambda x:x['initial_optimizer_hashes'].__setitem__('actor','different_moments'))
            with self.assertRaisesRegex(ValueError,'optimizer states differ'):
                launch.verify_warmup_pairs(protocol,root)

    def test_copied_cell_report_is_rejected_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory(dir=launch.HERE,prefix='pair_fixture_') as directory:
            root=Path(directory);protocol=fixtures(root)
            mutate_and_reseal(root,'seed0_hole','report',lambda x:x.__setitem__('condition','full'))
            with self.assertRaisesRegex(ValueError,'cell/protocol identity differs'):
                launch.verify_warmup_pairs(protocol,root)


if __name__=='__main__':unittest.main()
