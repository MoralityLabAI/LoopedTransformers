"""Private local story model transport; no oracle or host labels imported."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from experiments.nan_prime20.wire import OrderedSlots,stream_chunks,sse_bytes
from .common import public_rows,eligible_rows,verify,read,sha,write,append
from .engine import Engine


def main():
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('bf16','nf4'),required=True);p.add_argument('--bind',required=True)
    p.add_argument('--client-ip',required=True);p.add_argument('--port',type=int,default=18764);p.add_argument('--snapshot');p.add_argument('--compactification-root')
    p.add_argument('--bpd-root');p.add_argument('--out',required=True);p.add_argument('--mock',action='store_true');p.add_argument('--mock-limit',type=int,default=2);a=p.parse_args()
    address=ipaddress.ip_address(a.bind);client=ipaddress.ip_address(a.client_ip)
    if a.bind in ('0.0.0.0','::') or not address.is_private or not client.is_private:raise ValueError('One local private adapter only')
    token=os.environ['NAN_STORY23_TOKEN'];assert len(token)>=32
    root=Path(__file__).parent;cfg=read(root/'config.json');public=public_rows();dev=eligible_rows(a.backend,software=a.mock)
    slots=OrderedSlots(dev); assert len(dev) in (72,144)
    if not a.mock:verify()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cache=out/'HF_cache';cache.mkdir()
    os.environ['HF_HOME']=str(cache.resolve());os.environ['HF_HUB_OFFLINE']='1';os.environ.pop('TRANSFORMERS_CACHE',None)
    started=time.monotonic();engine=None
    try:
        if not a.mock:engine=Engine(a,cfg,out,public)
    except Exception as error:
        write(out/'receipt.json',dict(status='failed_model_loading',failure_type=type(error).__name__,requests=0,mock_only=False,sealed_test_accessed=False));raise
    write(out/'manifest.json',dict(experiment=cfg['experiment'],backend=a.backend,mock_only=a.mock,pid=os.getpid(),
        source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json') if not a.mock else None,
        host_labels_opened=False,registered_contexts=144,eligible_contexts=len(dev),maximum_generated_tokens=64,KV_cache=False,early_parse_stop='first complete anchored packet',
        head_device=str(engine.head.weight.device) if engine else None,CPU_threads=engine.torch.get_num_threads() if engine else None,sealed_test_accessed=False))
    completed=[];lock=threading.Lock();failures=[];limit=a.mock_limit if a.mock else len(dev)
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
            try:body=json.loads(self.rfile.read(length))
            except (ValueError,TypeError) as error:return self.reply(400,{'error':str(error)})
            if not lock.acquire(blocking=False):return self.reply(409,{'error':'one request at a time'})
            ordinal=slots.next_index
            try:
                if time.monotonic()-started>=cfg['maximum_backend_seconds']:return self.reply(429,{'error':'registered time budget reached'})
                try:ordinal,key,row=slots.claim(body,a.backend,cfg)
                except (ValueError,TypeError,KeyError) as error:return self.reply(400,{'error':str(error)})
                identifier='nan-story23-'+a.backend+'-'+str(ordinal)
                result=dict(text='MOCK_ONLY',finish_reason='length',usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2)) if a.mock else engine.generate(row,ordinal)
                write(out/f'request_{ordinal:04d}.json',dict(completion_id=identifier,prompt_sha256=key,source_id=row['id'],source_ordinal=ordinal,mock_only=a.mock,request=body,**result))
                completed.append(identifier);slots.commit(row['id']);append(out/'events.jsonl',dict(event='request_closed',requests=len(completed),source_id=row['id'],mock_only=a.mock))
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
            requests=len(completed),attempted_prompts=len(slots.attempted),failure_types=failures,registered_contexts=144,eligible_contexts=len(dev),elapsed_seconds=time.monotonic()-started,
            source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json') if not a.mock else None,host_labels_opened=False,sealed_test_accessed=False,BPD_transfer_attempted=False)
        if engine:
            receipt|=engine.close()
        write(out/'receipt.json',receipt)


if __name__=='__main__':main()
