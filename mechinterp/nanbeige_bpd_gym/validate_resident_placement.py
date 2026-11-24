"""Random small-model test of increased decoder residency and unchanged head."""
import argparse
import os
from pathlib import Path
import time
import torch
from experiments.nan_bpd01.backend import NativeLoop
from experiments.nan_comp02.head import OperatorHead
from experiments.nan_gym05.observe import TowerObserver
from experiments.nan_fit07.io import sha,write
from mechinterp.dark_fiber_gyms import arg03_nanbeige as NB
from .resident_head import mount,retain_decoder_layers,check_parity


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError('Preserve prior engineering evidence')
    cache=Path('runs/resident_placement_tokenizer_cache');cache.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(2026112008)
    torch.cuda.set_per_process_memory_fraction(.10);torch.cuda.reset_peak_memory_stats()
    model,_=NB.build_nanbeige_model_and_loop(Path(a.snapshot),exec_device=torch.device('cpu'),dtype=torch.bfloat16,n_gpu_layers=22,random_init=True,
        config_overrides={'num_hidden_layers':22,'hidden_size':64,'intermediate_size':96,'num_attention_heads':4,
            'num_key_value_heads':2,'head_dim':16,'kv_channels':16})
    loop=NativeLoop(model,'cuda:0',quantized=False);observer=TowerObserver(loop,(5,11,17,21))
    tokens=torch.randint(0,100,(19,)).tolist()
    before2,physical,before1=observer.forward({'engineering':tokens},capture=True)
    cpu=OperatorHead(model.lm_head.weight,dict(I=[1],N=[2],S=[3],R=[4]),chunk_rows=8192)
    retain_decoder_layers(loop,4);resident=mount(loop,cpu)
    after2,changed,after1=observer.forward({'engineering':tokens},capture=True)
    assert all(torch.equal(x,y) for x,y in zip(before1+before2,after1+after2))
    assert physical.keys()==changed.keys() and all(torch.equal(physical[k],changed[k]) for k in physical)
    checks=[check_parity(cpu,resident,h) for h in (after1[0],after2[0])];observer.close()
    write(out,{'status':'passed','evidence_class':'random 22-layer width-64 residency engineering; not pretrained inference',
        'pretrained_weights_loaded':False,'decoder_resident_prefix_before':1,'decoder_resident_prefix_after':4,
        'boundary_states_bit_exact':True,'physical_state_vectors_bit_exact':len(physical),'head_checks':checks,
        'resident_source_sha256':sha(Path(__file__).with_name('resident_head.py')),'validation_source_sha256':sha(__file__),
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-started,'sealed_test_accessed':False})
    print({'status':'passed','physical_vectors_bit_exact':len(physical),'full_vocabulary_logits_checked':sum(r['full_vocabulary_logits_checked'] for r in checks),
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'pretrained_weights_loaded':False})


if __name__=='__main__':main()
