"""Record software without creating another CUDA context."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    packages={}
    for name in ['torch','transformers','bitsandbytes','accelerate','numpy','safetensors','tokenizers','huggingface-hub','matplotlib']:
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    result={'python_executable':sys.executable,'python_version':sys.version,'platform':platform.platform(),'packages':packages,
        'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','PYTORCH_CUDA_ALLOC_CONF','HF_HUB_OFFLINE']},
        'environment_scope':'this metadata command, not the separate model processes','created_cuda_context':False,'sealed_test_accessed':False}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
