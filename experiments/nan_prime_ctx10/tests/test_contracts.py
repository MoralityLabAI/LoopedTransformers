import ast
from pathlib import Path
import unittest
from experiments.nan_prime_ctx10.common import public_rows,read
from experiments.nan_prime06.wire import prompt_sha,validate_request


class ContextPrimeContracts(unittest.TestCase):
    def test_complete_roster_and_exact_unconstrained_token_cap(self):
        _,dev=public_rows();self.assertEqual(len(dev),288)
        self.assertEqual(len({r['root'] for r in dev}),4)
        self.assertEqual(sum(r['copy_control'] for r in dev),144)
        cfg=read(Path('experiments/nan_prime_ctx10/config.json'));source=dev[0];prompts={prompt_sha(source['prompt']):source}
        body=dict(model='nanbeige-451ed48c-nf4',messages=[dict(role='user',content=source['prompt'])],temperature=0,top_p=1,max_tokens=1)
        self.assertEqual(validate_request(body,prompts,'nf4',cfg)[1],source)
        for cap in (0,2,16):
            with self.assertRaises(ValueError):validate_request(body|{'max_tokens':cap},prompts,'nf4',cfg)

    def test_model_server_and_loader_do_not_import_gold_or_evaluator(self):
        root=Path('experiments/nan_prime_ctx10')
        for name in ('server.py','common.py'):
            tree=ast.parse((root/name).read_text())
            strings=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
            self.assertNotIn('fixtures/host_labels.jsonl',strings)
            imported=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            self.assertFalse(any(n.endswith(('.taskset','.validate_linux','.report')) for n in imported))
        # The transport rejects a source after its first actual attempt, including failures.
        text=(root/'server.py').read_text();self.assertIn('if key in attempted:',text)
        self.assertLess(text.index('attempted.add(key)'),text.index('engine.generate(row,ordinal)'))


if __name__=='__main__':unittest.main()
