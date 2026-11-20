"""Optional verifiers.v1 adapter; no cloud calls, model loads or training on import."""
from __future__ import annotations
# API source reviewed at PrimeIntellect-ai/verifiers commit below. Install into
# an isolated environment; never upgrade the source-locked 3050 inference venv.
VERIFIERS_API_COMMIT='ed70053b6a7a4079d6976e03bdcd7d420356d594'
import verifiers.v1 as vf
from .task_rows import load_rows,score_reply

class NanbeigeDiagnosticData(vf.TaskData):
    answer: int|str|list[str]
    kind: str
    root_id: str
    arm: str
    source_manifest_sha256: str
    gold_assistance: bool

class NanbeigeDiagnosticTask(vf.Task[NanbeigeDiagnosticData]):
    @vf.reward
    async def exact(self,trace:vf.Trace)->float:
        trace.info['root_id']=self.data.root_id;trace.info['arm']=self.data.arm
        trace.info['source_manifest_sha256']=self.data.source_manifest_sha256
        trace.info['endpoint']='complete_final_reply_grammar; not the registered local prefix-stop endpoint'
        return score_reply(trace.last_reply or '',self.data.kind,self.data.answer)

class NanbeigeDiagnosticConfig(vf.TasksetConfig):
    experiment_root: str='experiments/nan_diag04'
    split: str='development'
    component: str='free'

class NanbeigeDiagnosticTaskset(vf.Taskset[NanbeigeDiagnosticTask,NanbeigeDiagnosticConfig]):
    def load(self)->list[NanbeigeDiagnosticTask]:
        rows=load_rows(self.config.experiment_root,split=self.config.split,component=self.config.component)
        return [NanbeigeDiagnosticTask(NanbeigeDiagnosticData(**{k:r[k] for k in ('id','prompt','answer','kind','root_id','arm','source_manifest_sha256','gold_assistance')}),self.config.task) for r in rows]
