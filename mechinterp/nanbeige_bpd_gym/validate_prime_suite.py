"""Real pinned SDK construction/reward checks; not a model benchmark."""
import argparse
import asyncio
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
from .prime_suite import NanbeigeSuiteConfig,NanbeigeSuiteTaskset
from .prime_taskset import VERIFIERS_API_COMMIT

async def validate(scope):
    direct=json.loads(importlib.metadata.distribution('verifiers').read_text('direct_url.json'))
    if direct.get('vcs_info',{}).get('commit_id')!=VERIFIERS_API_COMMIT:raise ValueError('Installed SDK revision mismatch')
    tasks=list(NanbeigeSuiteTaskset(NanbeigeSuiteConfig(scope=scope)))
    for i,task in enumerate(tasks):
        assert task.data.idx==i
        answer=task.data.private_label['answer'];text=answer if type(answer) is str else json.dumps(answer)
        trace=SimpleNamespace(last_reply=text,info={})
        assert await task.correct(trace)==1 and trace.info['root_id']==task.data.root_id
        trace.last_reply='Unparseable answer'
        assert await task.correct(trace)==0
    return {'status':'passed','scope':'official pinned SDK task construction/iteration and direct reward checks; no model, agent, runtime or hosted rollout',
        'sdk_commit':VERIFIERS_API_COMMIT,'sdk_version':importlib.metadata.version('verifiers'),'dataset_scope':scope,
        'development_main_rows':len(tasks),'root_groups':len({t.data.root_id for t in tasks}),
        'suite_rows':dict(Counter(t.data.suite for t in tasks)),'private_gold_injected_into_prompt_by_adapter':False,
        'paid_API_calls':0,'sealed_test_accessed':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=('public','local'),default='public');p.add_argument('--out',required=True);a=p.parse_args()
    result=asyncio.run(validate(a.scope));path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result))

if __name__=='__main__':main()
