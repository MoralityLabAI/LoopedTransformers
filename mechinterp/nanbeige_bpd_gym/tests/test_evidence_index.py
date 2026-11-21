import json
import unittest
from unittest.mock import patch
from mechinterp.nanbeige_bpd_gym.evidence_index import build
from mechinterp.nanbeige_bpd_gym.evidence_index import STUDIES,study,digest,extract,task_scores,render


class EvidenceIndexContracts(unittest.TestCase):
    def test_failed_fitting_gate_is_not_presented_as_a_queued_fit(self):
        rows=extract('FIT07',dict(backends={'nf4':dict(status='competence_gate_failed',
            independent_audit=dict(baseline=dict(allow_fits=False)),cells=[dict(passes=1,status='pending'),dict(passes=2,status='pending')])}))
        data=dict(source_commit='a'*40,evidence_signature='s',studies=[dict(study='FIT07',endpoint='fit',published_report=True,
            report_markdown='NAN_FIT07_PROGRESS.md',rows=rows)])
        text=render(data);self.assertIn('unrun: fixed fitting gate closed',text);self.assertNotIn('T1 pending',text)

    def test_index_commit_does_not_create_an_infinite_publication_loop(self):
        with patch('mechinterp.nanbeige_bpd_gym.evidence_index.study',side_effect=lambda spec,loader:dict(study=spec[0],report_sha256='r',source_manifest_sha256='s')):
            first=build('a'*40,loader=lambda *args:None,producer_sha256='producer1')
            second=build('b'*40,loader=lambda *args:None,producer_sha256='producer1')
            changed=build('c'*40,loader=lambda *args:None,producer_sha256='producer2')
        self.assertEqual(first['evidence_signature'],second['evidence_signature'])
        self.assertNotEqual(first['evidence_signature'],changed['evidence_signature'])

    def test_source_mismatch_and_unsealed_assertion_are_not_published(self):
        spec=next(s for s in STUDIES if s[0]=='KV12');seal=b'{"registered_cells": 1}'
        data=dict(source_manifest_sha256='0'*64,sealed_test_accessed=False,backends={})
        def loader(path,optional):return seal if path.endswith('SOURCE_MANIFEST.json') else json.dumps(data).encode()
        with self.assertRaises(ValueError):study(spec,loader)
        data['source_manifest_sha256']=digest(seal);data['sealed_test_accessed']=True
        with self.assertRaises(ValueError):study(spec,loader)

    def test_registration_without_execution_stays_explicitly_unscored(self):
        spec=next(s for s in STUDIES if s[0]=='KV12');seal=b'{}'
        result=study(spec,lambda path,optional:seal if path.endswith('SOURCE_MANIFEST.json') else None)
        self.assertFalse(result['published_report']);self.assertEqual(result['rows'],[])
        self.assertEqual(result['status'],'registered_no_published_execution_report')

    def test_group_accuracy_keeps_copy_CAL_and_loops_separate(self):
        rows=[dict(split='development',loop=2,copy_control=False,contexts=3,correct=1),
            dict(split='development',loop=1,copy_control=False,contexts=3,correct=3),
            dict(split='calibration',loop=2,copy_control=False,contexts=20,correct=20),
            dict(split='development',loop=2,copy_control=True,contexts=3,correct=2)]
        result=task_scores(rows,'contexts','copy_control',development=True,loop2=True)
        self.assertEqual(result,dict(main=dict(correct=1,scored_cases=3),copy=dict(correct=2,scored_cases=3)))

    def test_partial_SDK_coverage_is_not_imputed_and_extra_text_is_dropped(self):
        backend=dict(status='terminal_with_missing_coverage',registered_contexts=10,neural_requests=5,successful_official_episodes=4,
            missing_graded_contexts=6,orphan_or_unbound_neural_replies=1,physical_prefix_vectors_byte_exact=200,
            groups=[dict(copy_control=False,graded_contexts=3,correct=1),dict(copy_control=True,graded_contexts=1,correct=1)],
            raw_reply='PRIVATE_SENTINEL',token_ids=[42])
        result=extract('PRIME-CTX10',dict(backends={'nf4':backend}))[0]
        self.assertEqual(result['scores']['main'],dict(correct=1,scored_cases=3));self.assertEqual(result['missing_graded_contexts'],6)
        self.assertNotIn('PRIVATE_SENTINEL',json.dumps(result));self.assertNotIn('token_ids',result)
        backend['missing_graded_contexts']=5
        with self.assertRaises(ValueError):extract('PRIME-CTX10',dict(backends={'nf4':backend}))

    def test_cache_diagnostic_never_becomes_task_accuracy(self):
        values=dict(status='partial_evidence',registered_contexts=130,closed_context_records=32,prefill_contexts_byte_exact=32,
            reference_token_positions=112,free_argmax_disagreements=3,accuracy=.99,raw_outputs='PRIVATE_SENTINEL')
        r=extract('KV11',dict(backends={'bf16':values}))[0]
        self.assertNotIn('accuracy',r);self.assertNotIn('scores',r);self.assertNotIn('PRIVATE_SENTINEL',json.dumps(r))


if __name__=='__main__':unittest.main()
