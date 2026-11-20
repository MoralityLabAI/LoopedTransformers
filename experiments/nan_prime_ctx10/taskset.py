"""Official SDK taskset; host labels never enter the model-serving process."""
from __future__ import annotations
from typing import Any
from pathlib import Path
import verifiers.v1 as vf
from mechinterp.nanbeige_bpd_gym.audit_native import final_answer
from .common import public_rows,CTX_SEAL,read,rows,sha

__all__=['NanbeigeContextTaskset']


class ContextData(vf.TaskData):
    source_root:str
    context_budget:int
    position:str
    task_kind:str
    world:str
    copy_control:bool
    answer_schema:dict[str,Any]
    private_answer:str
    source_manifest_sha256:str


class ContextTask(vf.Task[ContextData]):
    async def validate(self,runtime:vf.Runtime)->bool:
        return final_answer(self.data.private_answer,self.data.answer_schema)==self.data.private_answer and final_answer('Invalid label',self.data.answer_schema) is None

    @vf.reward
    async def correct(self,trace:vf.Trace)->float:
        answer=final_answer(trace.last_reply or '',self.data.answer_schema)
        trace.info|={'source_root':self.data.source_root,'context_budget':self.data.context_budget,'position':self.data.position,
            'task_kind':self.data.task_kind,'world':self.data.world,'copy_control':self.data.copy_control,
            'source_manifest_sha256':self.data.source_manifest_sha256,'format_valid':answer is not None,
            'endpoint':'one unconstrained native loop-2 token, complete official final reply; separate from PRIME06 multi-token endpoint'}
        return float(answer==self.data.private_answer)


class ContextConfig(vf.TasksetConfig):
    split:str='development'


class NanbeigeContextTaskset(vf.Taskset[ContextTask,ContextConfig]):
    def load(self)->list[ContextTask]:
        if self.config.split!='development':raise ValueError('Only the registered CTX08 development split is supported')
        _,public=public_rows();root=Path('experiments/nan_ctx08');seal=read(root/'SOURCE_MANIFEST.json');file=root/'fixtures/host_labels.jsonl'
        assert sha(file)==seal['host_only_files_sha256']['fixtures/host_labels.jsonl']
        labels={r['id']:r['answer'] for r in rows(file)}
        return [ContextTask(ContextData(id=r['id'],prompt=r['prompt'],source_root=r['root'],context_budget=r['context_budget'],
            position=r['position'],task_kind=r['task'],world=r['world'],copy_control=r['copy_control'],answer_schema=r['schema'],
            private_answer=labels[r['id']],source_manifest_sha256=CTX_SEAL),self.config.task) for r in public]
