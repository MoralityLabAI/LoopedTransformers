"""Real Lean induction/rcases continuation tests, excluded from research scores."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from bounded_multigoal_search import LeanCompiler,solve
from resources import ResourceGuard
from train_local import write_json

class SmokeRanker:
    def __init__(self,kind):self.kind=kind
    def rank(self,goal,candidates):
        if self.kind=='induction':
            desired='rfl' if 'case ' in goal else 'induction xs'
        else:
            desired='exact fc12_left' if 'fc12_left' in goal else 'rcases h with ⟨fc12_left, fc12_right⟩'
        assert any(s==desired for _,s in candidates),(desired,candidates)
        return sorted(candidates,key=lambda c:c[1]!=desired)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lean',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    sources={
      'induction':'import Init\nnamespace RAM02BranchTest\ntheorem toy (xs : List Nat) : xs.length = xs.length := by\n  induction xs\n  rfl\n  rfl\nend RAM02BranchTest\n',
      'rcases':'import Init\nnamespace RAM02BranchTest\ntheorem toy (P Q : Prop) (h : P ∧ Q) : P := by\n  rcases h with ⟨fc12_left, fc12_right⟩\n  exact fc12_left\nend RAM02BranchTest\n'}
    initial={'induction':'xs : List Nat\n⊢ xs.length = xs.length',
             'rcases':'P Q : Prop\nh : P ∧ Q\n⊢ P'}
    results={}
    with ResourceGuard(args.out/'resources.jsonl',torch) as guard:
        compiler=LeanCompiler(args.lean,guard)
        for kind,source in sources.items():
            row=solve(source,'toy',[initial[kind]],SmokeRanker(kind),args.out/kind,compiler)
            assert row['accepted'],row
            if kind=='induction':
                assert [r['goal_count'] for r in row['calls'] if r['probe']]==[2,1,0],row
            final=row['final_certificate']
            replay=compiler.compile_source(Path(final['source_file']).read_text(encoding='utf8'),False,30.,args.out/(kind+'-replay'))
            assert replay['accepted'] and replay['source_sha256']==final['source_sha256']
            row['replay']=replay
            results[kind]=row
        results['resources']=guard.receipt()
    results['counted_as_regression_proofs']=False
    write_json(args.out/'branch_smoke.json',results)
    print('RAM02_REAL_INDUCTION_RCASES_SEARCH_VERIFIED',flush=True)

if __name__=='__main__':main()
