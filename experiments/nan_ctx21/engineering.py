"""MOCK_ONLY worker I/O/stop fixture; no torch import, CUDA call or model tensors."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
from .common import ROOT, read, sha, write


def main():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    import numpy as np
    from . import worker
    output=ROOT/'receipts/orchestration_mock_v1.json'; assert not output.exists()
    workspace=Path.cwd().resolve(); parent=workspace/'runs/ctx21_engineering_fixture'; parent.mkdir(parents=True,exist_ok=True)
    cfg=read(ROOT/'config.json'); ids=[1]*4045; idhash=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    widths=dict(residual_after_layer=3072,down_proj_input=10752,down_proj_output=3072,o_proj_input=6144,o_proj_output=3072)
    outcomes=[]
    with tempfile.TemporaryDirectory(prefix='mock_only_',dir=parent) as temporary:
        task=Path(temporary).resolve(); assert task.is_relative_to(parent.resolve()) and parent.resolve().is_relative_to(workspace)
        static=task/'source'; (static/'fixtures').mkdir(parents=True)
        public=[dict(id=f'MOCK_ONLY:{i}',root='MOCK_ONLY',split='calibration',context_budget=4096,position='middle',world='A',
            arm='main',graph_records=52,target_graph_index=26,prompt='MOCK_ONLY') for i in range(624)]
        (static/'fixtures/public.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in public),encoding='utf-8')
        write(static/'SOURCE_MANIFEST.json',dict(MOCK_ONLY=True))
        snapshot=task/'snapshot'; snapshot.mkdir(); (snapshot/'tokenizer.json').write_text('MOCK_ONLY',encoding='utf-8')
        write(static/'fixtures/tokenizer_receipt.json',dict(contexts={r['id']:dict(tokens=len(ids),ids_sha256=idhash) for r in public},
            tokenizer_files_sha256={'tokenizer.json':sha(snapshot/'tokenizer.json')}))

        class InjectedOOM(RuntimeError):
            pass

        class NoGrad:
            def __enter__(self):
                return self
            def __exit__(self,*args):
                return False

        torch=types.SimpleNamespace(set_num_threads=lambda n:None,manual_seed=lambda n:None,get_num_threads=lambda:1,
            inference_mode=NoGrad,cuda=types.SimpleNamespace(OutOfMemoryError=InjectedOOM,
                set_per_process_memory_fraction=lambda n:None,reset_peak_memory_stats=lambda:None,
                max_memory_allocated=lambda:0,max_memory_reserved=lambda:0))

        class Vector:
            def __init__(self,width):
                self.width=width
            def numpy(self):
                return np.zeros(self.width,dtype=np.float32)

        class Observer:
            calls=0
            closes=0
            def __init__(self,loop,layers):
                pass
            def forward(self,sequences,*,capture):
                Observer.calls+=1; assert capture and sequences=={'context':ids}
                if Observer.calls==3:
                    raise InjectedOOM('MOCK_ONLY injected third-prefix capacity stop')
                values={('context',t,l,s):Vector(width) for t in (1,2) for l in (5,11,17,21) for s,width in widths.items()}
                return [object()],values,[object()]
            def close(self):
                Observer.closes+=1

        class Head:
            weight=types.SimpleNamespace(device='cuda:0',dtype='torch.bfloat16')
            def __init__(self,*args,**kwargs):
                pass
            def score(self,h):
                return dict(vocabulary_size=166144,free_top_token_id=1,free_top_token_logit=0.,full_log_normalizer=1.,
                    selected_logits={},class_logprobs=dict(I=-4.,N=-4.,S=-4.,R=-4.),grammar_mass=4*np.exp(-4.))

        loop=types.SimpleNamespace(stage_layer=None,n_gpu_layers=22,
            base=types.SimpleNamespace(embed_tokens=types.SimpleNamespace(weight=types.SimpleNamespace(device='cpu'))))
        tokenizer=types.SimpleNamespace(tok=types.SimpleNamespace(decode=lambda *args,**kwargs:'A'))
        parity=lambda *args:dict(full_vocabulary_logits_checked=166144,vocabulary_logits_bit_exact=True,all_scorer_fields_exact=True)
        imports={
            'torch':torch,
            'experiments.nan_bpd01.backend':types.SimpleNamespace(load_backend=lambda *args,**kwargs:(loop,types.SimpleNamespace(weight=None),tokenizer,None)),
            'experiments.nan_comp02.head':types.SimpleNamespace(OperatorHead=Head),
            'experiments.nan_comp03.prompt':types.SimpleNamespace(aliases=lambda tok:None),
            'experiments.nan_gym05.common':types.SimpleNamespace(prompt_ids=lambda *args:ids),
            'experiments.nan_gym05.observe':types.SimpleNamespace(TowerObserver=Observer),
            'experiments.nan_recomp09.common':types.SimpleNamespace(target_digest=lambda loop:'MOCK_ONLY_UNCHANGED'),
            'mechinterp.dark_fiber_gyms.arg03_nanbeige':types.SimpleNamespace(assert_oracle_isolated=lambda:None),
            'mechinterp.nanbeige_bpd_gym.resident_head':types.SimpleNamespace(mount=lambda *args:Head(),retain_decoder_layers=lambda *args:None,check_parity=parity)}
        for label,seconds,expected in [('injected_oom',14400,'capacity_stopped'),('between_prefix_budget',0,'budget_reached')]:
            runfolder=task/label; config=dict(cfg,maximum_backend_seconds=seconds)
            (static/'config.json').write_text(json.dumps(config),encoding='utf-8')
            argv=['worker','--backend','nf4','--snapshot',str(snapshot),'--compactification-root',str(task)]
            with patch.dict(sys.modules,imports),patch.object(worker,'ROOT',static),patch.object(worker,'RUNS',runfolder),\
                patch.object(worker,'verify',lambda:dict(adapter_commit='MOCK_ONLY')),patch.object(worker.subprocess,'check_output',lambda *args,**kwargs:'MOCK_ONLY\n'),\
                patch('experiments.nan_ranksite13.resources.driver_sample',lambda:dict(MOCK_ONLY=True)),patch.object(sys,'argv',argv):
                worker.main()
            receipt=read(runfolder/'nf4/receipt.json'); events=[json.loads(v) for v in (runfolder/'nf4/events.jsonl').read_text().splitlines()]
            assert receipt['status']==expected and events[-1]['event']=='worker_receipt_closed'
            casefiles=list((runfolder/'nf4').glob('case_*.json')); statefiles=list((runfolder/'nf4').glob('case_*_states.npz'))
            count=2 if label=='injected_oom' else 0
            assert receipt['completed_contexts']==len(casefiles)==len(statefiles)==count
            assert [v['ordinal'] for v in events if v['event']=='case_closed']==list(range(count))
            for file in statefiles:
                with np.load(file,allow_pickle=False) as z:
                    assert len(z.files)==40 and all(np.isfinite(z[name]).all() for name in z.files)
            outcomes.append(dict(MOCK_ONLY=True,condition=label,terminal_status=expected,closed_cases=count,closed_archives=count,
                untouched_remaining_slots=624-count,closed_receipt_sha256=sha(runfolder/'nf4/receipt.json')))
        assert Observer.calls==3 and Observer.closes==1
        assert task.resolve().is_relative_to(parent.resolve()),'Verify workspace containment before TemporaryDirectory cleanup'
    write(output,dict(status='passed',evidence_class='MOCK_ONLY actual worker file/orchestration/stop path; zero neural calls, model tensors, CUDA calls or real grades.',
        outcomes=outcomes,worker_forward_stub_invocations=Observer.calls,physical_archive_stub_vectors_checked=80,
        producer_sha256=sha(__file__),worker_sha256=sha(ROOT/'worker.py'),common_sha256=sha(ROOT/'common.py'),
        numerical_CPU_threads=1,priority='BelowNormal',pretrained_model_loaded=False,CUDA_used=False,sealed_test_accessed=False))
    print(dict(status='passed',MOCK_ONLY=True,closed_cases=2,capacity_stop_preserved=True,budget_stop_preserved=True))


if __name__=='__main__':
    main()
