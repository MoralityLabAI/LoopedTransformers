"""Wider-suite tasks exported by nanbeige_bpd_suite; no paid/model calls."""
from __future__ import annotations
from typing import Any
import json
import verifiers.v1 as vf
from .suite_rows import load_suite_rows,score_suite_reply,SOURCE_SEAL

__all__=['NanbeigeSuiteTaskset']

class NanbeigeSuiteData(vf.TaskData):
    root_id:str
    suite:str
    answer_schema:dict[str,Any]
    private_label:dict[str,Any]
    source_manifest_sha256:str

class NanbeigeSuiteTask(vf.Task[NanbeigeSuiteData]):
    async def validate(self,runtime:vf.Runtime)->bool:
        """Official gold check of the grader; no agent/model invocation."""
        answer=self.data.private_label['answer'];text=answer if type(answer) is str else json.dumps(answer)
        good=score_suite_reply(text,self.data.answer_schema,self.data.private_label)
        bad=score_suite_reply('Unparseable answer',self.data.answer_schema,self.data.private_label)
        return good['correct']==1 and bad['correct']==0

    @vf.reward
    async def correct(self,trace:vf.Trace)->float:
        scores=score_suite_reply(trace.last_reply or '',self.data.answer_schema,self.data.private_label)
        trace.info|={'root_id':self.data.root_id,'suite':self.data.suite,'source_manifest_sha256':self.data.source_manifest_sha256,
            'endpoint':'complete_final_reply; not pooled with native prefix-stop results',
            'format_valid':scores['format_valid'],'reference_exact':scores['reference_exact']}
        if 'valid_givens_preserving_solution' in scores:trace.info['valid_sudoku_solution']=scores['valid_givens_preserving_solution']
        return float(scores['correct'])

class NanbeigeSuiteConfig(vf.TasksetConfig):
    repository_root:str|None=None
    scope:str='public'
    split:str='development'
    suite:str='all'
    include_copy_controls:bool=False

class NanbeigeSuiteTaskset(vf.Taskset[NanbeigeSuiteTask,NanbeigeSuiteConfig]):
    def load(self)->list[NanbeigeSuiteTask]:
        cfg=self.config;values=load_suite_rows(cfg.repository_root,scope=cfg.scope,split=cfg.split,suite=cfg.suite,include_copy_controls=cfg.include_copy_controls)
        return [NanbeigeSuiteTask(NanbeigeSuiteData(id=r['id'],prompt=r['prompt'],root_id=r['root_id'],suite=r['suite'],
            answer_schema=r['schema'],private_label=r['private_label'],source_manifest_sha256=SOURCE_SEAL),cfg.task) for r in values]
