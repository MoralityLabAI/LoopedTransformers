"""Local-only native checkpoint endpoint for the official Prime null harness.

The worker opens registered prompt/schema rows, never host gold or task graders.
All requests, replies, tokens and physical states stay in local runs/.
"""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from experiments.nan_gym05.common import read,rows,sha,write
from .wire import prompt_sha,validate_request,stream_chunks,sse_bytes


class Engine:
    def __init__(self,args,cfg,out):
        import torch
        from experiments.nan_bpd01.backend import load_backend
        from experiments.nan_comp02.head import OperatorHead
        from experiments.nan_comp03.prompt import aliases
        from experiments.nan_gym05.observe import TowerObserver
        self.torch=torch;self.cfg=cfg;self.out=out;self.args=args
        torch.manual_seed(2026112006);torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction']);torch.cuda.reset_peak_memory_stats()
        self.loop,old,self.tok,_=load_backend(args.compactification_root,args.snapshot,args.backend,chunk_rows=cfg['vocabulary_chunk_rows'])
        self.head=OperatorHead(old.weight,aliases(self.tok),chunk_rows=cfg['vocabulary_chunk_rows'])
        self.observer=TowerObserver(self.loop,cfg['layers_zero_based'])

    def generate(self,row,ordinal):
        import numpy as np
        from experiments.nan_gym05.common import prompt_ids
        cfg=self.cfg;torch=self.torch;started=time.monotonic()
        with torch.inference_mode():
            ids=prompt_ids(self.tok,self.args.snapshot,row['prompt']);assert len(ids)<=cfg['maximum_prompt_tokens']
            cap=cfg['maximum_'+row['schema']['type']+'_tokens'];generated=[];scores=[];states={};index=[]
            for step in range(cap):
                h,observed,_=self.observer.forward({'request':ids+generated},capture=step==0)
                if step==0:
                    for i,((_,phase,layer,site),tensor) in enumerate(observed.items()):
                        key='v'+str(i);states[key]=tensor.numpy();index.append({'loop':phase,'layer_zero_based':layer,'site':site,'array_key':key})
                score=self.head.score(h[0]);token=score['free_top_token_id'];generated.append(token);scores.append(score)
                if token==self.tok.tok.eos_token_id:break
            ended=generated[-1]==self.tok.tok.eos_token_id
            text=self.tok.tok.decode(generated[:-1] if ended else generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            file=self.out/f'request_{ordinal:04d}_states.npz';np.savez_compressed(file,**states)
            torch.cuda.empty_cache()
            return {'text':text,'finish_reason':'stop' if ended else 'length','initial_prompt_ids':ids,'generated_ids':generated,
                'scores':scores,'elapsed_seconds':time.monotonic()-started,'state_file':file.name,'state_file_sha256':sha(file),'state_index':index,
                'usage':{'prompt_tokens':len(ids),'completion_tokens':len(generated),'total_tokens':len(ids)+len(generated)}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('bf16','nf4'),required=True)
    p.add_argument('--bind',required=True);p.add_argument('--client-ip',required=True);p.add_argument('--port',type=int,default=18761)
    p.add_argument('--snapshot');p.add_argument('--compactification-root');p.add_argument('--out',required=True)
    p.add_argument('--cohort',default='runs/nan_gym05/cohort/public.jsonl');p.add_argument('--mock',action='store_true');a=p.parse_args()
    # Bind one loopback/virtual-adapter address, never all interfaces or a public IP.
    address=ipaddress.ip_address(a.bind);client=ipaddress.ip_address(a.client_ip)
    if a.bind in ('0.0.0.0','::') or not address.is_private or not client.is_private:raise ValueError('Local adapter only')
    token=os.environ['NAN_PRIME_LOCAL_TOKEN'];assert len(token)>=24
    root=Path(__file__).parent;cfg=read(root/'config.json');source=read('experiments/nan_gym05/SOURCE_MANIFEST.json')
    if sha('experiments/nan_gym05/SOURCE_MANIFEST.json')!=cfg['source_manifest_sha256'] or sha(a.cohort)!=source['combined_public_sha256']:raise ValueError('Source registration changed')
    public=[r for r in rows(a.cohort) if r['split']==cfg['split']];assert len(public)==cfg['contexts_per_backend']
    prompts={};multiplicity={}
    for r in public:
        assert set(r)=={'id','root','split','suite','prompt','schema','publication'}
        key=prompt_sha(r['prompt'])
        if key in prompts and prompts[key]['schema']!=r['schema']:raise ValueError('Ambiguous schema for the same model input')
        prompts[key]=r;multiplicity[key]=multiplicity.get(key,0)+1
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);engine=None
    if not a.mock:
        seal=read(root/'SOURCE_MANIFEST.json')
        for name,h in seal['files_sha256'].items():
            if sha(root/name)!=h:raise ValueError('Prime condition source changed: '+name)
        for name,h in seal['dependencies_sha256'].items():
            if sha(name)!=h:raise ValueError('Prime dependency changed: '+name)
        if sha(Path(a.snapshot)/'tokenizer_config.json')!=source['tokenizer_config_sha256']:raise ValueError('Tokenizer changed')
        try:engine=Engine(a,cfg,out)
        except Exception as e:
            write(out/'receipt.json',{'backend':a.backend,'status':'failed_model_loading','requests':0,
                'failure_type':type(e).__name__,'failure_message':str(e),'sealed_test_accessed':False,'BPD_transfer_attempted':False})
            raise
    started=time.monotonic();lock=threading.Lock();counts={};completed=[]
    write(out/'manifest.json',{'experiment':cfg['experiment'],'backend':a.backend,'mock_only':a.mock,'pid':os.getpid(),
        'configuration':cfg,'source_registration_sha256':None if a.mock else sha(root/'SOURCE_MANIFEST.json'),
        'prompt_cohort_sha256':sha(a.cohort),'unique_prompts':len(prompts),'host_labels_opened':False,
        'model_endpoint':'local Windows virtual adapter; official Linux Prime subprocess/null harness',
        'early_parse_stop':False,'KV_cache':False,'sealed_test_accessed':False,'BPD_transfer_attempted':False})
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass  # Prompts and auth never enter console/access logs.
        def reply(self,status,value):
            b=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
        def allowed(self):
            return self.client_address[0] in ('127.0.0.1',a.client_ip,a.bind) and self.headers.get('Authorization')=='Bearer '+token
        def do_GET(self):
            if not self.allowed():return self.reply(403,{'error':'local authorization required'})
            if self.path=='/health':return self.reply(200,{'ready':True,'mock':a.mock,'backend':a.backend,'completed_requests':len(completed)})
            if self.path=='/v1/models':return self.reply(200,{'object':'list','data':[{'id':'nanbeige-451ed48c-'+a.backend,'object':'model','created':0,'owned_by':'local-pinned-checkpoint'}]})
            return self.reply(404,{'error':'unknown local path'})
        def do_POST(self):
            if not self.allowed():return self.reply(403,{'error':'local authorization required'})
            if self.path=='/shutdown':
                self.reply(200,{'stopping_after_inflight_request':True})
                threading.Thread(target=server.shutdown,daemon=True).start();return
            if self.path!='/v1/chat/completions':return self.reply(404,{'error':'unsupported local path'})
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<1048576:return self.reply(400,{'error':'invalid request size'})
            try:
                body=json.loads(self.rfile.read(length));key,row=validate_request(body,prompts,a.backend,cfg)
            except (ValueError,TypeError,KeyError) as e:return self.reply(400,{'error':str(e)})
            if not lock.acquire(blocking=False):return self.reply(409,{'error':'one local request at a time'})
            try:
                if time.monotonic()-started>cfg['maximum_backend_seconds']:return self.reply(429,{'error':'registered backend time budget reached'})
                if counts.get(key,0)>=multiplicity[key]:return self.reply(409,{'error':'registered prompt multiplicity exhausted; no replay cache'})
                ordinal=len(completed);identifier='nan-prime06-'+a.backend+'-'+str(ordinal)
                result={'text':'MOCK_ONLY','finish_reason':'stop','usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}} if a.mock else engine.generate(row,ordinal)
                write(out/f'request_{ordinal:04d}.json',{'completion_id':identifier,'prompt_sha256':key,'request':body,'mock_only':a.mock,**result})
                counts[key]=counts.get(key,0)+1;completed.append(identifier)
                write(out/'progress.json',{'requests':len(completed),'mock_only':a.mock,'elapsed_seconds':time.monotonic()-started,'prompt_counts':counts})
                if body.get('stream'):
                    b=sse_bytes(stream_chunks(identifier,body['model'],result['text'],result['usage'],result['finish_reason']))
                    self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
                else:self.reply(200,{'id':identifier,'object':'chat.completion','created':0,'model':body['model'],
                    'choices':[{'index':0,'message':{'role':'assistant','content':result['text']},'finish_reason':result['finish_reason']}],'usage':result['usage']})
                print(json.dumps({'completed_request':ordinal,'mock':a.mock,'tokens':result['usage']['completion_tokens']}),flush=True)
                if not a.mock and len(completed)==len(public):threading.Thread(target=server.shutdown,daemon=True).start()
            except Exception as e:
                write(out/f'failure_{len(completed):04d}.json',{'type':type(e).__name__,'message':str(e)})
                self.reply(500,{'error':'local generation failed; preserved for inspection'})
            finally:lock.release()
    server=ThreadingHTTPServer((a.bind,a.port),Handler);server.daemon_threads=False
    print(json.dumps({'ready':True,'mock_only':a.mock,'backend':a.backend,'allowed_contexts':len(public)}),flush=True)
    deadline=threading.Timer(cfg['maximum_backend_seconds'],server.shutdown);deadline.daemon=True;deadline.start()
    try:server.serve_forever(poll_interval=1)
    finally:
        deadline.cancel();server.server_close()
        receipt={'backend':a.backend,'mock_only':a.mock,'status':'completed' if len(completed)==len(public) else 'stopped_with_incomplete_coverage',
            'requests':len(completed),'registered_contexts':len(public),'elapsed_seconds':time.monotonic()-started,
            'host_labels_opened':False,'sealed_test_accessed':False,'BPD_transfer_attempted':False}
        if engine is not None:
            receipt|={'peak_cuda_allocated_bytes':engine.torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':engine.torch.cuda.max_memory_reserved(),
                'head_remained_cpu':engine.head.weight.device.type=='cpu'}
            engine.observer.close()
        write(out/'receipt.json',receipt)


if __name__=='__main__':main()
