"""Real SDK discovery and every host reward; no model or hosted execution."""
import argparse
import asyncio
from collections import Counter
import importlib.metadata as metadata
import json
from pathlib import Path
from types import SimpleNamespace
from .taskset import NanbeigeContextTaskset,ContextConfig
from .common import SDK_COMMIT,write


async def validate():
    direct=json.loads(metadata.distribution('verifiers').read_text('direct_url.json'));assert direct['vcs_info']['commit_id']==SDK_COMMIT
    from verifiers.v1.utils.loaders import taskset_class,load_taskset,environment_class
    module='nanbeige_context_suite';assert taskset_class(module) is NanbeigeContextTaskset
    assert environment_class(module).__name__=='SingleAgentEnv'
    tasks=list(load_taskset(ContextConfig(id=module)));assert len(tasks)==288
    for index,task in enumerate(tasks):
        assert task.data.idx==index
        for text,expected in [(task.data.private_answer,1),('Invalid label',0),(task.data.private_answer+' extra',0)]:
            trace=SimpleNamespace(last_reply=text,info={});assert await task.correct(trace)==expected
    for split in ('test','calibration','heldout','unknown'):
        try:list(NanbeigeContextTaskset(ContextConfig(split=split)))
        except ValueError:pass
        else:raise AssertionError('Non-development source scope was accepted')
    return dict(status='passed',sdk_commit=SDK_COMMIT,sdk_version=metadata.version('verifiers'),official_plugin_discovery=True,
        development_contexts=288,source_root_families=len({t.data.source_root for t in tasks}),
        task_counts=dict(Counter(t.data.task_kind for t in tasks)),copy_controls=sum(t.data.copy_control for t in tasks),
        direct_correct_invalid_and_trailing_text_reward_checks_per_context=3,non_development_scopes_rejected=True,
        model_rollouts=0,paid_API_calls=0,sealed_test_accessed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();r=asyncio.run(validate());write(Path(a.out),r);print(r)


if __name__=='__main__':main()
