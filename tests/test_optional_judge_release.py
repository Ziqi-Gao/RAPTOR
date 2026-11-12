"""Offline schema/identity checks; synthetic fixtures are never judge results."""
import json
from pathlib import Path
import tempfile
import unittest
from raptor.steering.judge_six_metrics import (audit_delivery, digest, prepare, verify_prepared)
from raptor.steering.judge_six_protocol import METHODS, ProtocolError


class OptionalJudgeTests(unittest.TestCase):
    def fixture(self, root):
        delivery=root/'delivery';delivery.mkdir()
        pair=dict(model_id='test/model',task='STSA',direction='neutral_to_negative',prompt_id=0,generation_seed=100001)
        pair_id=digest(tuple(pair.values()))
        identity=dict(checkpoint_revision='synthetic',tokenizer_revision='synthetic',chat_template_sha256='0'*64,decoding_config=dict(max_new_tokens=1))
        rows=[dict(**pair,record_id=digest([pair_id,method]),pair_id=pair_id,method=method,
                   prompt_text='synthetic prompt',raw_output='synthetic output',source_identity=identity) for method in METHODS]
        (delivery/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return delivery

    def test_prepare_is_offline_and_identity_guarded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);delivery=self.fixture(root)
            records,pairs,audit=audit_delivery(delivery,production=False)
            self.assertEqual(len(records),5);self.assertEqual(len(pairs),1)
            result=prepare(delivery,root/'run',production=False)
            self.assertEqual(result['status'],'OFFLINE_PREPARED_JUDGE_NOT_RUN')
            verify_prepared(root/'run',check_sources=True)
            self.assertFalse((root/'run'/'units').exists())
            with self.assertRaises(ProtocolError):audit_delivery(delivery,production=True)
            path=delivery/'records.jsonl';path.write_text(path.read_text()+'{}\n')
            with self.assertRaises(ProtocolError):verify_prepared(root/'run',check_sources=True)


if __name__=='__main__':unittest.main()
