from pathlib import Path
import argparse
import time
from runtime import Session,ResourceGuard,certify,compile_source
import ram03_core as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--lean',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    source='import Init\nnamespace RAM03Test\n/-- Documentation must attach to the theorem. -/\ntheorem toy (P Q : Prop) (h : P) (k : Q) : P ∧ Q := by\n  constructor\n  exact h\n  exact k\nend RAM03Test\n'
    with ResourceGuard(a.out/'resources.jsonl') as guard:
        with Session(a.lean,source,'toy',a.out/'server',guard) as session:
            assert len(session.initial['goals'])==1
            one=session.query(['constructor']);two=session.query(['constructor','exact h']);three=session.query(['constructor','exact h','exact k'])
            assert [len(x['goals']) for x in (one,two,three)]==[2,1,0]
            assert not session.query(['exact missing'])['accepted']
            assert c.signature(session.query(['constructor'])['goals'])==c.signature(one['goals'])
            start=time.perf_counter()
            for _ in range(20):assert session.query(['constructor','exact h','exact k'])['accepted']
            rate=20/(time.perf_counter()-start)
        final=certify(a.lean,source,'toy',['constructor','exact h','exact k'],a.out/'final',guard,10)
        assert final['accepted'],final
        replay=compile_source(a.lean,Path(final['source_file']).read_text(encoding='utf8'),a.out/'replay',guard,10)
        assert replay['accepted'] and replay['source_sha256']==final['source_sha256']
        bad=source.replace('theorem toy (P Q : Prop) (h : P) (k : Q) : P ∧ Q := by','axiom poisoned : False\ntheorem toy : False := by')
        rejected=certify(a.lean,bad,'toy',['exact poisoned'],a.out/'bad',guard,10)
        assert rejected['returncode']==0 and not rejected['accepted']
        c.write_json(a.out/'smoke.json',dict(startup_seconds=session.startup_seconds,evaluations_per_second=rate,
          final=final,replay=replay,bad=rejected,resources=guard.receipt(),counted_as_research_proofs=False))
        print('RAM03_SERVER_SMOKE',session.startup_seconds,rate,flush=True)
if __name__=='__main__':main()
