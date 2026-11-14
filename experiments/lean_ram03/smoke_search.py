from pathlib import Path
import argparse
from runtime import ResourceGuard
from search import search
import ram03_core as c

class ForcedBranches:
    def rank(self,state,pool):
        goal=state['goals'][0].split('⊢',1)[1].strip()
        desired='constructor' if '∧' in goal else ('exact h' if goal=='P' else 'exact k')
        assert any(s==desired for _,s in pool)
        return sorted(pool,key=lambda item:item[1]!=desired)

def main():
    p=argparse.ArgumentParser();p.add_argument('--lean',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    original='import Init\nnamespace RAM03SearchTest\ntheorem toy (P Q : Prop) (h : P) (k : Q) : P ∧ Q := by\n  constructor\n  exact h\n  exact k\nend RAM03SearchTest\n'
    with ResourceGuard(a.out/'resources.jsonl') as guard:
        result=search(a.lean,original,'toy',ForcedBranches(),a.out/'evidence',guard,
          dict(seconds=30,evaluations=48,transitions=8,request_seconds=3,max_goals=8))
        assert result['accepted'],result
        assert result['actions']==['constructor','exact h','exact k']
        assert [r['after_goals'] for r in result['calls']]==[2,1,0]
        result['counted_as_research_proofs']=False
        c.write_json(a.out/'smoke_search.json',result)
        print('RAM03_FULL_SEARCH_REPLAY_SMOKE_PASSED',result['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
