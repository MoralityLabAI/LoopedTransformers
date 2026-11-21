"""Actual pinned SDK discovery and all144 independent-oracle reward checks."""
import argparse
import asyncio
import importlib.metadata as metadata
import json
import os
from types import SimpleNamespace
from .common import SDK_COMMIT,ROOT,sha,write


async def validate():
    os.nice(19)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    from verifiers.v1.utils.loaders import taskset_class,load_taskset,environment_class
    from .taskset import NanbeigeStoryworldTaskset,StoryConfig
    from .oracle import grade
    direct=json.loads(metadata.distribution('verifiers').read_text('direct_url.json'));assert direct['vcs_info']['commit_id']==SDK_COMMIT
    module='nanbeige_storyworld_suite';assert taskset_class(module) is NanbeigeStoryworldTaskset
    assert environment_class(module).__name__=='SingleAgentEnv'
    tasks=list(load_taskset(StoryConfig(id=module,software_only=True)));assert len(tasks)==144
    for index,task in enumerate(tasks):
        assert task.data.idx==index
        gold=task.data.gold_packet;canonical=gold['canonical_trace']
        probes=[canonical,'MOCK_ONLY',canonical+' extra',' '*513+canonical,canonical.replace('FINAL='+gold['final_room'],'FINAL=R'+str((int(gold['final_room'][1])+1)%8))]
        for text in probes:
            trace=SimpleNamespace(last_message=SimpleNamespace(content=text),last_reply=text.strip(),info={})
            expected=grade(text,task.data.schema_packet,gold)
            assert await task.trace_exact(trace)==float(expected['trace_exact'])
            assert trace.info['endpoints']==expected
    for split in ('test','heldout','calibration','unknown'):
        try:list(NanbeigeStoryworldTaskset(StoryConfig(split=split,software_only=True)))
        except ValueError:pass
        else:raise AssertionError('Unregistered scope accepted')
    return dict(status='passed',sdk_commit=SDK_COMMIT,sdk_version=metadata.version('verifiers'),official_plugin_discovery=True,
        contexts=144,roots=12,reward_checks=720,raw_untrimmed_512_character_bound=True,non_development_scopes_rejected=True,
        model_rollouts=0,paid_API_calls=0,sealed_test_accessed=False,
        source_sha256={n:sha(ROOT/n) for n in ('taskset.py','parser.py','oracle.py','common.py','validate_linux.py')},
        entry_point_sha256=sha('nanbeige_storyworld_suite.py'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    result=asyncio.run(validate());write(a.out,result);print(result)
