"""Selector failure modes and exact preservation of original per-case checks."""
import ast
import unittest
from .common import ROOT, OLD, select_prefix
from experiments.nan_ctx21.audit import grade


class Tests(unittest.TestCase):
    def fixture(self):
        journal = [dict(event='case_closed', ordinal=i, completed_contexts=i+1,
                        context_budget=4096 if i<312 else 8192) for i in range(378)]
        queue = [dict(event='worker_terminal', backend='bf16', exit_code=4294967295),
                 dict(event='audit_terminal', backend='bf16', exit_code=1)]
        outcome = dict(BF16_worker_ended=True, original_terminal_audit='failed_missing_normal_receipt',
                       original_receipt_synthesized=False, closed_native_cases_preserved=378,
                       original_worker_exit_code=4294967295, closed_4K=312, closed_8K=66)
        return [journal, queue, outcome, False]

    def test_closed_prefix(self):
        self.assertEqual(select_prefix(*self.fixture()), 378)

    def test_reject_noncontiguous_or_duplicate(self):
        for ordinal in (1, 379):
            args = self.fixture(); args[0][-1]['ordinal'] = ordinal
            with self.assertRaises(AssertionError): select_prefix(*args)

    def test_reject_live_success_or_synthesized_receipt(self):
        for change in ('live', 'success', 'receipt', 'synthesized', 'native_receipt_event'):
            args = self.fixture()
            if change=='live': args[2]['BF16_worker_ended']=False
            if change=='success': args[1][-1]['exit_code']=0
            if change=='receipt': args[3]=True
            if change=='synthesized': args[2]['original_receipt_synthesized']=True
            if change=='native_receipt_event': args[0].append(dict(event='worker_receipt_closed'))
            with self.assertRaises(AssertionError): select_prefix(*args)

    def test_case_checks_AST_identical(self):
        def block(path):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            return next(v for v in ast.walk(tree) if isinstance(v,ast.For) and
                        isinstance(v.target,ast.Tuple) and [getattr(x,'id',None) for x in v.target.elts]==['ordinal','source'])
        self.assertEqual(ast.dump(block(OLD/'audit.py')), ast.dump(block(ROOT/'audit.py')))

    def test_original_bounded_parser(self):
        for text in ('A', '**B**.', '`P`', ' [C] '): self.assertIsNotNone(grade(text))
        for text in ('Answer: A', 'AA', 'a', '**A** extra', 'A'*65): self.assertIsNone(grade(text))


if __name__=='__main__': unittest.main()
