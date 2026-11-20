import re
import unittest
from pathlib import Path
from experiments.nan_fit07.io import read,rows
from experiments.nan_ctx08.sources import build,LETTERS


def independent_answer(text):
    records={k:(n,l) for k,n,l in re.findall(r'^Record (R\d{6}): next=(R\d{6}); label=([A-P])\.$',text,re.M)}
    q=re.search(r'Question: label\((next\()?((?:R)\d{6})\)?\)',text)
    join,key=q.groups()
    if join:key=records[key][0]
    return records[key][1],records


class ContextSourceTests(unittest.TestCase):
    def test_registered_291_sources_and_token_bands(self):
        root=Path(__file__).resolve().parents[1];public=rows(root/'fixtures/public.jsonl')
        labels={r['id']:r for r in rows(root/'fixtures/host_labels.jsonl')};tokens=read(root/'fixtures/tokenizer_receipt.json')['contexts']
        self.assertEqual(len(public),291);self.assertEqual(len({r['root'] for r in public if r['split']=='development'}),4)
        for r in public:
            self.assertEqual(independent_answer(r['prompt'])[0],labels[r['id']]['answer'])
            self.assertTrue(r['context_budget']-96<=tokens[r['id']]['tokens']<=r['context_budget'])

    def test_all_positions_tasks_and_worlds_are_structurally_correct(self):
        for position in (.1,.5,.9):
            for task in ('lookup','join'):
                a,ga=build(2026112308,41,position,task,'a',False)
                b,gb=build(2026112308,41,position,task,'b',False)
                answer,records=independent_answer(a);other,changed=independent_answer(b)
                self.assertEqual((answer,other),(ga['answer'],gb['answer']));self.assertNotEqual(answer,other)
                self.assertEqual(len(records),41);self.assertEqual(set(records),{v[0] for v in records.values()})
                self.assertEqual([k for k in records if records[k]!=changed[k]],[ga['changed_key']])
                self.assertTrue(all(v[1] in LETTERS for v in records.values()))
                copy,gc=build(2026112308,41,position,task,'a',True)
                self.assertIn('supplied label: '+gc['answer']+'\nAnswer:',copy)
                self.assertEqual(independent_answer(copy)[1],records)


if __name__=='__main__':unittest.main()
