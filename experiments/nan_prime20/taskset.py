"""Official SDK host taskset; gold values are never loaded by the model server."""
from pathlib import Path
import verifiers.v1 as vf
from experiments.nan_comp14.parser import label
from .common import COMP14_SEAL, public_rows, read, rows, sha

__all__ = ['NanbeigeCompetenceTaskset']


class CompetenceData(vf.TaskData):
    source_root: str
    arm: str
    world: str
    private_answer: str
    source_manifest_sha256: str


class CompetenceTask(vf.Task[CompetenceData]):
    async def validate(self, runtime: vf.Runtime) -> bool:
        return label(self.data.private_answer) == self.data.private_answer and label('Invalid label') is None

    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        value = label(trace.last_reply or '')
        trace.info |= dict(source_root=self.data.source_root, arm=self.data.arm, world=self.data.world,
            source_manifest_sha256=self.data.source_manifest_sha256, format_valid=value is not None,
            endpoint='one unconstrained native loop2 token, complete official final reply; no bounded-reply substitution')
        return float(value == self.data.private_answer)


class CompetenceConfig(vf.TasksetConfig):
    split: str = 'development'


class NanbeigeCompetenceTaskset(vf.Taskset[CompetenceTask, CompetenceConfig]):
    def load(self) -> list[CompetenceTask]:
        if self.config.split != 'development':
            raise ValueError('Only the registered COMP14 development sources are supported')
        _, public = public_rows(); parent = Path('experiments/nan_comp14'); file = parent / 'fixtures/host_labels.jsonl'
        assert sha(file) == read(parent / 'SOURCE_MANIFEST.json')['host_files_sha256']['fixtures/host_labels.jsonl']
        labels = {r['id']: r['answer'] for r in rows(file)}
        return [CompetenceTask(CompetenceData(id=r['id'], prompt=r['prompt'], source_root=r['root'], arm=r['arm'],
            world=r['world'], private_answer=labels[r['id']], source_manifest_sha256=COMP14_SEAL), self.config.task) for r in public]
