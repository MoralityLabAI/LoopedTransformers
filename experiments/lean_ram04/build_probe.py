"""Build the RAM-04 narrow-import helper with the shared resource guard."""
import argparse
from pathlib import Path
import ram04_core as c
from runtime import ResourceGuard,guarded_process
def main():
    p=argparse.ArgumentParser();p.add_argument('--lean',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    with ResourceGuard(a.out/'resources.jsonl') as guard:
        result=guarded_process([a.lean,'-j','2','-DElab.async=false','ProbeServer.lean','-o','ProbeServer.olean'],a.out/'build.log',guard,60,cwd=Path(__file__).parent)
        assert result['returncode']==0 and not result['timed_out'],result
        c.write_json(a.out/'build.json',dict(result=result,lean_executable_sha256=c.digest(a.lean),resources=guard.receipt()))
    print('RAM04_HELPER_BUILT',flush=True)
if __name__=='__main__':main()
