"""Official SDK host-side grading, isolated from the model process."""
import verifiers.v1 as vf
from .common import ROOT,eligible_rows,read,rows,sha
from .parser import parse


class StoryData(vf.TaskData):
    source_root:str
    arm:str
    world:str
    mode:str
    schema_packet:dict
    gold_packet:dict
    source_manifest_sha256:str


def endpoints(text,schema,gold):
    packet=parse(text,schema)
    if packet is None:return dict(format_valid=0,final_correct=0,intermediate_correct=[0,0,0],path_correct=0,trace_exact=0)
    intermediate=[int(a==b) for a,b in zip(packet['intermediate_rooms'],gold['intermediate_rooms'])]
    final=int(packet['final_room']==gold['final_room']);path=int(packet['path']==gold['path'])
    return dict(format_valid=1,final_correct=final,intermediate_correct=intermediate,path_correct=path,
        trace_exact=int(bool(final and path and all(intermediate))))


class StoryTask(vf.Task[StoryData]):
    async def validate(self,runtime:vf.Runtime)->bool:
        return parse(self.data.gold_packet['canonical_trace'],self.data.schema_packet) is not None

    @vf.reward
    async def trace_exact(self,trace:vf.Trace)->float:
        # SDK last_reply strips whitespace. Bound and grade the raw assistant
        # node, so outer trimming cannot rescue a response over 512 characters.
        values=endpoints(trace.last_message.content or '',self.data.schema_packet,self.data.gold_packet)
        trace.info|=dict(source_root=self.data.source_root,arm=self.data.arm,world=self.data.world,mode=self.data.mode,
            source_manifest_sha256=self.data.source_manifest_sha256,endpoints=values,
            endpoint='complete bounded trace, not internal reasoning faithfulness')
        return float(values['trace_exact'])


class StoryConfig(vf.TasksetConfig):
    split:str='development'
    backend:str='nf4'
    software_only:bool=False


class NanbeigeStoryworldTaskset(vf.Taskset[StoryTask,StoryConfig]):
    def load(self)->list[StoryTask]:
        if self.config.split!='development' or self.config.backend not in ('nf4','bf16'):
            raise ValueError('Only registered fresh story development sources are supported')
        file=ROOT/'fixtures/host_labels.jsonl';manifest=ROOT/'SOURCE_MANIFEST.json'
        if not self.config.software_only:
            seal=read(manifest);assert sha(file)==seal['critical_files_sha256']['fixtures/host_labels.jsonl']
        labels={r['id']:r for r in rows(file)}
        public=eligible_rows(self.config.backend,software=self.config.software_only)
        return [StoryTask(StoryData(id=r['id'],prompt=r['prompt'],source_root=r['root'],arm=r['arm'],world=r['world'],
            mode=r['mode'],schema_packet=r['schema'],gold_packet=labels[r['id']],
            source_manifest_sha256=sha(manifest) if manifest.exists() else 'SOFTWARE_ONLY_UNFROZEN'),self.config.task) for r in public]
