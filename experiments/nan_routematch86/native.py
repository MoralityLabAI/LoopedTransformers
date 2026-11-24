"""Deliver a phase-1 MLP write edit on a duplicate B2 row."""

from experiments.nan_radial76.native import WriterReaderCapture, WriterReaderInstrument
from experiments.nan_reader71.native import ReaderInstrument
from experiments.nan_vocab82.capture import Observer
from experiments.nan_vocab82.heads import verify_head
from experiments.nan_writer77.pipeline import scoped_trial
from mechinterp.nanbeige_bpd_gym.chunked_cache import ChunkedTwoLoopCache

from .geometry import ARMS


class MlpWriterInstrument(WriterReaderInstrument):
    def __init__(self,loop,replay):
        ReaderInstrument.__init__(self,loop,0,replay)
        self.writer=loop.n-1
        module=loop.layers[self.writer] if self.writer<loop.n_gpu_layers else loop.stage_layer
        if module is None:
            raise ValueError("The final MLP writer has no execution layer")
        self.handles.append(module.mlp.down_proj.register_forward_hook(self.write))
        self.writer_edit=None


class MlpWriterCapture(WriterReaderCapture):
    def __init__(self,loop,head,*,chunk=128,replay=None):
        self.cache=ChunkedTwoLoopCache(loop,prefill_chunk=chunk,observe_layers=(),
                                      instrument_factory=lambda model:MlpWriterInstrument(model,replay))
        self.instrument=self.cache.instrument
        self.head=head


def observed_trial(capture, ids, clean, writes, bare, reduce, classes, check):
    if set(writes)!=set(ARMS):
        raise ValueError("All three matched MLP writer arms are required")
    mapping={"active":"learned","original_seeded":"cal_static","matched_seeded":"seeded",
             "active_k_rescue":"learned_k_rescue"}
    observer=Observer(capture,reduce,trial=True)
    result=scoped_trial(observer,ids,clean,{k:writes[v] for k,v in mapping.items()
                                            if k!="active_k_rescue"},bare,check)
    modes=("active","active_k_rescue","original_seeded","matched_seeded")
    maxima={}
    for key,(rescued,rows) in zip(modes,observer.rows,strict=True):
        if rescued != (key=="active_k_rescue"):
            raise ValueError("MLP observer attached the rescue to the wrong prefill")
        for observed,native in zip(rows,result["arms"][key],strict=True):
            verify_head(observed,native,classes,bare)
        maxima[mapping[key]]=rows
    if maxima["learned"][0]!=maxima["learned_k_rescue"][0]:
        raise ValueError("A phase-2 rescue changed phase-1 grammar maxima")
    return result | {"class_maxima":maxima,
                     "arms":{mapping[k]:v for k,v in result["arms"].items()},
                     "reader_l2":{mapping[k]:v for k,v in result["reader_l2"].items()}}
