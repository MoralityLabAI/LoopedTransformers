"""Real Lean tests for typed exact, rw and branching rcases; excluded from coverage."""
from pathlib import Path
import argparse
import ram04_core as c
from runtime import Session,ResourceGuard,certify,compile_source
CASES=[('application','(P Q : Prop) (h : P → Q) (hp : P) : Q',['exact h hp'],'exact_local'),
       ('rewrite','{α : Type} (f : α → Nat) (a b : α) (h : a = b) : f a = f b',['rw [h]'],'rw_local'),
       ('elimination','(P Q : Prop) (h : P ∨ Q) : Q ∨ P',['rcases h with r4a0 | r4b0','all_goals simp_all'],'rcases_local')]
def main():
    p=argparse.ArgumentParser();p.add_argument('--lean',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    with ResourceGuard(a.out/'resources.jsonl') as guard:
        for name,statement,actions,family in CASES:
            source=f'import Init\nnamespace Smoke\ntheorem {name} {statement} := by\n  sorry\nend Smoke\n'
            with Session(a.lean,source,name,a.out/name/'server',guard) as server:
                assert (family,actions[0]) in c.candidates(server.initial),(family,c.candidates(server.initial))
                first=server.query(actions[:1]);assert c.label(server.initial,first)>0
                if family=='rcases_local':assert len(first['goals'])==2
                assert not server.query(actions)['goals']
            final=certify(a.lean,source,name,actions,a.out/name/'final',guard,10.)
            replay=compile_source(a.lean,Path(final['source_file']).read_text(encoding='utf8'),a.out/name/'replay',guard,10.)
            assert final['accepted'] and replay['accepted'] and final['source_sha256']==replay['source_sha256']
            rows.append(dict(name=name,certificate=final,replay=replay))
        c.write_json(a.out/'smoke_families.json',dict(cases=rows,counted_as_research_proofs=False,resources=guard.receipt()))
    print('RAM04_FAMILY_COMPILES_AND_REPLAYS_PASSED',flush=True)
if __name__=='__main__':main()
