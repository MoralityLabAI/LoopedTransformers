"""Official pinned SDK discovery/reward checks only; no model requests."""
import argparse
import asyncio
from collections import Counter
import importlib.metadata as metadata
import json
from types import SimpleNamespace
from .common import SDK_COMMIT, write
from .taskset import NanbeigeCompetenceTaskset, CompetenceConfig


async def validate():
    from verifiers.v1.utils.loaders import taskset_class, load_taskset, environment_class
    direct = json.loads(metadata.distribution('verifiers').read_text('direct_url.json'))
    assert direct['vcs_info']['commit_id'] == SDK_COMMIT
    module = 'nanbeige_competence_suite'; assert taskset_class(module) is NanbeigeCompetenceTaskset
    assert environment_class(module).__name__ == 'SingleAgentEnv'
    tasks = list(load_taskset(CompetenceConfig(id=module))); assert len(tasks) == 96
    for index, task in enumerate(tasks):
        assert task.data.idx == index
        for text, expected in [(task.data.private_answer, 1), ('Invalid label', 0), (task.data.private_answer + ' extra', 0)]:
            trace = SimpleNamespace(last_reply=text, info={}); assert await task.correct(trace) == expected
    for split in ('test', 'calibration', 'heldout', 'unknown'):
        try:
            list(NanbeigeCompetenceTaskset(CompetenceConfig(split=split)))
        except ValueError:
            pass
        else:
            raise AssertionError('Unregistered source scope was accepted')
    return dict(status='passed', sdk_commit=SDK_COMMIT, sdk_version=metadata.version('verifiers'), official_plugin_discovery=True,
        contexts=96, source_roots=16, arm_counts=dict(Counter(t.data.arm for t in tasks)),
        direct_reward_checks_per_context=3, non_development_scopes_rejected=True,
        model_rollouts=0, paid_API_calls=0, sealed_test_accessed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True); a = p.parse_args()
    result = asyncio.run(validate()); write(a.out, result); print(result)
