"""Offline setup/reward smoke test against the isolated pinned real SDK."""
import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
from .prime_taskset import NanbeigeDiagnosticConfig,NanbeigeDiagnosticTaskset,VERIFIERS_API_COMMIT

async def validate():
    count=0
    for component in ('free','scaffold','copy','select','table','coded','labelcopy','codecopy','erased','erased_repeat'):
        config=NanbeigeDiagnosticConfig(component=component)
        tasks=NanbeigeDiagnosticTaskset(config).load()
        assert len(tasks)==(32 if component in ('free','scaffold','copy') else 8 if component in ('erased','erased_repeat') else 16)
        for task in tasks:
            answer=task.data.answer
            correct=answer[0] if isinstance(answer,list) else answer
            text=f'[{correct}]' if task.data.kind=='integer' else str(correct)
            trace=SimpleNamespace(last_reply=text,info={})
            score=await task.exact(trace)
            assert score==(.5 if isinstance(answer,list) else 1.)
            assert trace.info['root_id']==task.data.root_id
            trace.last_reply='unparseable answer'
            assert await task.exact(trace)==0.
            count+=1
    return {'status':'passed','scope':'real pinned SDK offline task construction and reward methods; no runtime, agent or model rollout',
        'sdk_commit':VERIFIERS_API_COMMIT,'sdk_version':importlib.metadata.version('verifiers'),'development_task_rows':count,
        'full_reply_endpoint_distinct_from_local_prefix_stop':True,'paid_API_calls':0,'model_loads':0,'sealed_test_accessed':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();r=asyncio.run(validate())
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(r))

if __name__=='__main__':main()
