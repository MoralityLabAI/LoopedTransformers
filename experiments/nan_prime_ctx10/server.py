"""Local one-token model endpoint; no gold labels or evaluator imports."""
import argparse
import gc
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from experiments.nan_prime06.wire import prompt_sha,validate_request,stream_chunks,sse_bytes
from .common import public_rows,verify_source,read,sha,write,CTX_SEAL


class Engine:
    def __init__(self,a,cfg,out,public):
        import torch
        from experiments.nan_bpd01.backend import load_backend
        from experiments.nan_comp02.head import OperatorHead
        from experiments.nan_comp03.prompt import aliases
        from experiments.nan_gym05.observe import TowerObserver
        from mechinterp.nanbeige_bpd_gym.resident_head import mount,retain_decoder_layers,check_parity
        from mechinterp.dark_fiber_gyms.arg03_nanbeige import assert_oracle_isolated
        self.torch=torch;self.a=a;self.cfg=cfg;self.out=out
        torch.set_num_threads(1);torch.manual_seed(cfg['source_seed']);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
        assert_oracle_isolated();self.loop,legacy,self.tok,_=load_backend(a.compactification_root,a.snapshot,a.backend,chunk_rows=cfg['vocabulary_chunk_rows'])
        if self.loop.stage_layer is not None:self.loop.stage_layer.requires_grad_(False).eval()
        if a.backend=='bf16':retain_decoder_layers(self.loop,cfg['BF16_resident_decoder_layers'])
        cpu=OperatorHead(legacy.weight,aliases(self.tok),chunk_rows=cfg['vocabulary_chunk_rows']);self.head=mount(self.loop,cpu);del legacy
        self.observer=TowerObserver(self.loop,cfg['layers_zero_based'])
        self.bindings=read('experiments/nan_ctx08/fixtures/tokenizer_receipt.json')
        assert sha(Path(a.snapshot)/'tokenizer_config.json')==self.bindings['tokenizer_config_sha256']
        cal=next(r for r in public if r['split']=='calibration' and r['context_budget']==1024)
        with torch.inference_mode():
            two,_,one=self.observer.forward({'context':self.ids(cal)},capture=False)
            checks=[check_parity(cpu,self.head,h) for h in (one[0],two[0])]
        write(out/'resident_head_parity.json',{'checks':checks,'source_manifest_sha256':sha(Path(__file__).parent/'SOURCE_MANIFEST.json')})
        del cpu,two,one;gc.collect()

    def ids(self,row):
        from experiments.nan_gym05.common import prompt_ids
        ids=prompt_ids(self.tok,self.a.snapshot,row['prompt']);bound=self.bindings['contexts'][row['id']]
        assert len(ids)==bound['tokens']<=self.cfg['maximum_prompt_tokens']
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==bound['initial_prompt_ids_sha256']
        return ids

    def generate(self,row,ordinal):
        import numpy as np
        t=self.torch;started=time.monotonic();ids=self.ids(row)
        with t.inference_mode():
            two,observed,one=self.observer.forward({'context':ids},capture=True)
            phases=[]
            for phase,h in enumerate((one[0],two[0]),1):
                score=self.head.score(h);token=score['free_top_token_id']
                phases.append({'loop':phase,'score':score,'free_token_text':self.tok.tok.decode([token],skip_special_tokens=False,clean_up_tokenization_spaces=False)})
        token=phases[-1]['score']['free_top_token_id'];ended=token==self.tok.tok.eos_token_id
        text='' if ended else phases[-1]['free_token_text'];arrays={};index=[]
        for i,((_,phase,layer,site),tensor) in enumerate(observed.items()):
            key='v'+str(i);arrays[key]=tensor.numpy();index.append(dict(loop=phase,layer_zero_based=layer,site=site,array_key=key))
        file=self.out/f'request_{ordinal:04d}_states.npz'
        with file.open('xb') as stream:np.savez_compressed(stream,**arrays)
        return dict(text=text,finish_reason='stop' if ended else 'length',initial_prompt_ids=ids,generated_ids=[token],
            phases=phases,state_file=file.name,state_file_sha256=sha(file),state_index=index,elapsed_seconds=time.monotonic()-started,
            usage=dict(prompt_tokens=len(ids),completion_tokens=1,total_tokens=len(ids)+1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('bf16','nf4'),required=True);p.add_argument('--bind',required=True)
    p.add_argument('--client-ip',required=True);p.add_argument('--port',type=int,default=18761);p.add_argument('--snapshot');p.add_argument('--compactification-root')
    p.add_argument('--out',required=True);p.add_argument('--mock',action='store_true');p.add_argument('--mock-limit',type=int,default=2);a=p.parse_args()
    address=ipaddress.ip_address(a.bind);client=ipaddress.ip_address(a.client_ip)
    if a.bind in ('0.0.0.0','::') or not address.is_private or not client.is_private:raise ValueError('One local private adapter only')
    token=os.environ['NAN_PRIME_CTX10_TOKEN'];assert len(token)>=32
    root=Path(__file__).parent;cfg=read(root/'config.json');public,dev=public_rows()
    prompts={prompt_sha(r['prompt']):r for r in dev};assert len(prompts)==len(dev)==288
    if not a.mock:verify_source(root)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cache=out/'HF_cache';cache.mkdir()
    os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    started=time.monotonic();engine=None
    try:
        if not a.mock:engine=Engine(a,cfg,out,public)
    except Exception as error:
        write(out/'receipt.json',dict(status='failed_model_loading',failure_type=type(error).__name__,requests=0,mock_only=False,sealed_test_accessed=False));raise
    write(out/'manifest.json',dict(experiment=cfg['experiment'],backend=a.backend,mock_only=a.mock,pid=os.getpid(),
        source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json') if not a.mock else None,CTX08_source_manifest_sha256=CTX_SEAL,
        host_labels_opened=False,registered_contexts=288,maximum_generated_tokens=1,KV_cache=False,early_parse_stop=False,
        head_device=str(engine.head.weight.device) if engine else None,CPU_threads=engine.torch.get_num_threads() if engine else None,sealed_test_accessed=False))
    attempted=set();completed=[];lock=threading.Lock();failures=[];limit=a.mock_limit if a.mock else 288
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def reply(self,status,value):
            b=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
        def allowed(self):return self.client_address[0] in ('127.0.0.1',a.bind,a.client_ip) and self.headers.get('Authorization')=='Bearer '+token
        def do_GET(self):
            if not self.allowed():return self.reply(403,{'error':'local authorization required'})
            if self.path=='/health':return self.reply(200,dict(ready=True,mock=a.mock,backend=a.backend,completed_requests=len(completed)))
            return self.reply(404,{'error':'unknown local path'})
        def do_POST(self):
            if not self.allowed():return self.reply(403,{'error':'local authorization required'})
            if self.path=='/shutdown':
                self.reply(200,{'stopping_after_inflight_request':True});threading.Thread(target=server.shutdown,daemon=True).start();return
            if self.path!='/v1/chat/completions':return self.reply(404,{'error':'unsupported local path'})
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<1048576:return self.reply(400,{'error':'invalid request size'})
            try:body=json.loads(self.rfile.read(length));key,row=validate_request(body,prompts,a.backend,cfg)
            except (ValueError,TypeError,KeyError) as error:return self.reply(400,{'error':str(error)})
            if not lock.acquire(blocking=False):return self.reply(409,{'error':'one request at a time'})
            try:
                if time.monotonic()-started>=cfg['maximum_backend_seconds']:return self.reply(429,{'error':'registered time budget reached'})
                if key in attempted:return self.reply(409,{'error':'source already attempted; no fresh retry or response replay'})
                attempted.add(key);ordinal=len(completed);identifier='nan-prime-ctx10-'+a.backend+'-'+str(ordinal)
                result=dict(text='MOCK_ONLY',finish_reason='length',usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2)) if a.mock else engine.generate(row,ordinal)
                write(out/f'request_{ordinal:04d}.json',dict(completion_id=identifier,prompt_sha256=key,mock_only=a.mock,request=body,**result))
                completed.append(identifier);write(out/'progress.json',dict(requests=len(completed),attempted_prompts=len(attempted),mock_only=a.mock))
                if body.get('stream'):
                    b=sse_bytes(stream_chunks(identifier,body['model'],result['text'],result['usage'],result['finish_reason']))
                    self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
                else:self.reply(200,dict(id=identifier,object='chat.completion',created=0,model=body['model'],
                    choices=[dict(index=0,message=dict(role='assistant',content=result['text']),finish_reason=result['finish_reason'])],usage=result['usage']))
                print({'completed_request':len(completed),'mock_only':a.mock},flush=True)
                if len(completed)==limit:threading.Thread(target=server.shutdown,daemon=True).start()
            except Exception as error:
                failures.append(type(error).__name__);write(out/f'failure_{ordinal:04d}.json',dict(type=type(error).__name__,message=str(error)))
                try:self.reply(500,{'error':'preserved local generation failure'})
                finally:threading.Thread(target=server.shutdown,daemon=True).start()
            finally:lock.release()
    server=ThreadingHTTPServer((a.bind,a.port),Handler);server.daemon_threads=False
    deadline=threading.Timer(max(1,cfg['maximum_backend_seconds']-(time.monotonic()-started)),server.shutdown);deadline.daemon=True;deadline.start()
    print(dict(ready=True,mock_only=a.mock,backend=a.backend),flush=True)
    try:server.serve_forever(poll_interval=1)
    finally:
        deadline.cancel();server.server_close()
        receipt=dict(backend=a.backend,mock_only=a.mock,status='completed' if len(completed)==limit else 'stopped_with_incomplete_coverage',
            requests=len(completed),attempted_prompts=len(attempted),failure_types=failures,registered_contexts=288,elapsed_seconds=time.monotonic()-started,
            source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json') if not a.mock else None,host_labels_opened=False,sealed_test_accessed=False,BPD_transfer_attempted=False)
        if engine:
            receipt|={'peak_cuda_allocated_bytes':engine.torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':engine.torch.cuda.max_memory_reserved(),'head_device':str(engine.head.weight.device)}
            engine.observer.close()
        write(out/'receipt.json',receipt)


if __name__=='__main__':main()
