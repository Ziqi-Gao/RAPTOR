"""Synthetic records use the real controls output/checkpoint schema; no model runs."""
import json
from pathlib import Path
import tempfile
import unittest
from raptor.steering.controls import atomic_json, digest, file_hash, summarize_trajectories
from raptor.steering.paper_judge import collect, requests, main


class PaperJudgeCollectionTests(unittest.TestCase):
    def make_run(self, root, method):
        directory = root / method
        config = dict(model='synthetic/model', dataset='STSA', direction='neutral_to_negative',
                      revision='synthetic-revision', dtype='float32', device='cpu', chat_template=True,
                      target_probability=.0001, mode='legacy_unit', activation_site='transformer_block_output',
                      decoding=dict(do_sample=True, max_new_tokens=2), generation_seeds=[1, 2],
                      reliability=dict(enabled=False, threshold=0, comparison='>='))
        identity = dict(protocol='measured_gcav_v1', config=config, probe_sha256=method+'-synthetic-sha',
                        prompts_sha256='synthetic-prompt-sha', source_sha256=dict(controls='synthetic-source'), active_layers=[1])
        probe = dict(model=config['model'], dataset='STSA', activation_site=config['activation_site'],
                     method=method, split_sha256='synthetic-shared-split', embedding_sha256='synthetic-shared-embedding')
        manifest = dict(identity=identity, probe_metadata=probe, prompt_count=1, record_count=2,
                        summary_unit='first-step layer-prompt')
        records = [dict(prompt_id=3, generation_seed=300000+seed, model_id=config['model'], task='STSA',
                        direction=config['direction'], method=method, prompt_text='Exact prompt\nwith whitespace. ',
                        original_output='Same original\ntext.', steered_output='' if seed == 2 else method+' output',
                        token_ids=[] if seed == 2 else [7,8], original_token_ids=[4,5],
                        trajectory=[dict(layer=1, step=0, probability_before=.5, probability_after=.0001,
                                         alpha_actual=-1., measurement_dtype='synthetic fixture', intervened=True, success=True)])
                   for seed in (1,2)]
        atomic_json(directory/'manifest.json', manifest)
        self.write_records(directory, records)
        return directory

    def write_records(self, directory, records):
        identity = json.loads((directory/'manifest.json').read_text())['identity']
        for record in records:
            key = {field: record[field] for field in ('prompt_id','generation_seed')}
            atomic_json(directory/'units'/(digest(key)+'.json'),
                        dict(status='complete', identity=dict(run_sha256=digest(identity),**key),
                             payload=record, payload_sha256=digest(record)))
        atomic_json(directory/'generations.json', records)
        atomic_json(directory/'control_summary.json', summarize_trajectories(records))

    def test_real_schema_collect_then_prepare_preserves_text_and_full_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); runs=[self.make_run(root,method) for method in ('singlelr','gcs')]
            before={path:file_hash(path) for run in runs for path in run.rglob('*.json')}
            output=root/'GENERATIONS.json'
            result=collect(runs,output)
            rows=json.loads(output.read_text())
            self.assertEqual((result['records'],result['pairs']),(6,2))
            self.assertEqual({row['generation_seed'] for row in rows},{300001,300002})
            self.assertEqual([row['method'] for row in rows].count('original'),2)
            self.assertEqual(sum(row['text']=='' for row in rows),2)
            self.assertTrue(all(row['prompt_text'].endswith(' ') for row in rows))
            self.assertEqual(before,{path:file_hash(path) for path in before})
            requests(rows,root/'judge')
            self.assertEqual(json.loads((root/'judge'/'request_identity.json').read_text())['pairs'],2)
            with self.assertRaisesRegex(ValueError,'overwrite'):collect(runs,output)
            with self.assertRaises(FileExistsError):requests(rows,root/'judge')
            rows[0]['text']='changed collected reference'
            atomic_json(output,rows)
            with self.assertRaisesRegex(ValueError,'source provenance'):
                main(['prepare','--generations',str(output),'--out',str(root/'second_judge')])

    def test_shared_original_disagreement_is_rejected_without_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); a=self.make_run(root,'singlelr');b=self.make_run(root,'gcs')
            records=json.loads((b/'generations.json').read_text());records[0]['original_output']='Changed reference'
            self.write_records(b,records)
            with self.assertRaisesRegex(ValueError,'shared Original'):collect([a,b],root/'out.json')
            self.assertFalse((root/'out.json').exists())

    def test_generation_identity_and_complete_unit_guards(self):
        for mismatch in ('revision','missing_record','corrupt_unit','full_seed','extra_unit'):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);a=self.make_run(root,'singlelr');b=self.make_run(root,'gcs')
                if mismatch=='revision':
                    manifest=json.loads((b/'manifest.json').read_text());manifest['identity']['config']['revision']='different'
                    atomic_json(b/'manifest.json',manifest)
                elif mismatch=='missing_record':
                    records=json.loads((b/'generations.json').read_text());atomic_json(b/'generations.json',records[:1])
                elif mismatch=='corrupt_unit':
                    unit=next((b/'units').glob('*.json'));value=json.loads(unit.read_text());value['payload']['steered_output']='changed'
                    atomic_json(unit,value)
                elif mismatch=='full_seed':
                    records=json.loads((b/'generations.json').read_text());records[0]['generation_seed']=1
                    atomic_json(b/'generations.json',records)
                else:atomic_json(b/'units'/'unindexed.json',dict(status='complete'))
                with self.assertRaises(ValueError):collect([a,b],root/'out.json')
                self.assertFalse((root/'out.json').exists())


if __name__=='__main__':unittest.main()
