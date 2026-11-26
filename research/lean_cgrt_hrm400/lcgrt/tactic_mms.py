"""Clock- and carry-aware MMS for the native byte-level Mathlib trainer.

Only the differentiated recursion is instrumented. Training-row IDs and
supervision stage distinguish identical tokens reached via different clocks.
No new graph edges, no verifier calls, no use of validation/test labels.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

# The local gym is normally launched from research/lean_cgrt_hrm400.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_gym.interp.mms import MultiSiteMemory, read_map_manifest, file_sha256


def active_rows(draw, carry, carried_ids):
    """Return the actual active train row and zero-based supervision stage."""
    draw = torch.as_tensor(draw, dtype=torch.long, device='cpu')
    if carry is None:
        if carried_ids is not None:
            raise ValueError('orphaned TRM carry IDs')
        return draw, torch.zeros_like(draw)
    if carried_ids is None or carried_ids.shape != draw.shape:
        raise ValueError('TRM carry requires its original row IDs')
    halted = carry['halted'].detach().cpu()
    stage = torch.where(halted, torch.zeros_like(draw), carry['steps'].detach().cpu())
    return torch.where(halted, draw, carried_ids), stage


class TacticMemory:
    def __init__(self, model, options, train_rows, max_depth):
        self.cfg = model.cfg
        self.train_rows = int(train_rows)
        self.max_depth = int(max_depth)
        c = self.cfg
        if self.train_rows < 1 or self.max_depth < 1:
            raise ValueError('empty train dataset or invalid depth')
        if c.arch == 'trm':
            blocks = {f'trm{i}': b for i,b in enumerate(model.trm_blocks)}
            self.stride = c.trm_n + 2
        elif c.arch in ('hrm','loop'):
            blocks = {'fast':model.fast}
            if c.arch == 'hrm':
                blocks['slow'] = model.slow
            self.stride = self.max_depth + 1
        else:
            raise ValueError('MMS requires a tied recurrent encoder')
        modules = {}
        for name, block in blocks.items():
            modules[f'{name}.ff'] = block.ff[-1]
            modules[f'{name}.att'] = block.att.o
            if getattr(block,'cross',None) is not None:
                modules[f'{name}.cross'] = block.cross.o
        names = list(modules) if options['sites']=='auto' else options['sites'].split(',')
        if not names or len(set(names)) != len(names) or not set(names).issubset(modules):
            raise ValueError('invalid Mathlib MMS site list')
        maps = read_map_manifest(options['map_manifest'],options['map_manifest_sha']) if options['map_manifest'] else {}
        self.memory = MultiSiteMemory({name:modules[name] for name in names},
            mib=options['mib'],mode=options['mode'],rank=options['rank'],
            topk=options['topk'],ratio=options['ratio'],refresh=options['refresh'],
            max_age=options['max_age'],schedule=options['schedule'],maps=maps)
        self.visit = options['visit']

    def capture(self, step, ids, depth=None, stage=None):
        ids = torch.as_tensor(ids,dtype=torch.long,device='cpu')
        if bool(((ids<0)|(ids>=self.train_rows)).any()):
            raise ValueError('MMS key outside train split')
        c = self.cfg
        name = self.memory.select(step)
        if c.arch == 'trm':
            if stage is None or stage.shape!=ids.shape or bool(((stage<0)|(stage>=c.trm_sup)).any()):
                raise ValueError('invalid supervision stage')
            keys = ids*c.trm_sup+stage
            visits = c.trm_n+1
        else:
            depth = c.fast_steps if depth is None else int(depth)
            if not 1<=depth<=self.max_depth:
                raise ValueError('depth outside registered MMS schedule')
            keys = ids*(self.max_depth+1)+depth
            visits = (depth+c.slow_every-1)//c.slow_every if name.startswith('slow.') else depth
        visit = visits//2 if self.visit<0 else self.visit
        if visit>=visits:
            raise ValueError('selected MMS visit outside current clock')
        return self.memory.sites[name].capture(keys,step,visit,visits,
            grad_only=c.arch=='trm',key_stride=self.stride)

    def state_dict(self):
        return self.memory.state_dict()

    def load_state_dict(self,state):
        self.memory.load_state_dict(state)

    def receipt(self):
        return self.memory.receipt()


def options_from_args(a):
    mib = float(getattr(a,'mms_mib',0))
    if mib==0:
        if getattr(a,'mms_map_manifest',''):
            raise ValueError('map manifest requires positive MMS budget')
        return None
    path = getattr(a,'mms_map_manifest','')
    return dict(mib=mib,mode=a.mms_mode,sites=a.mms_sites,schedule=a.mms_schedule,
        ratio=a.mms_ratio,rank=a.mms_rank,topk=a.mms_topk,refresh=a.mms_refresh,
        max_age=a.mms_max_age,visit=a.mms_visit,map_manifest=path,
        map_manifest_sha=file_sha256(path) if path else '')
