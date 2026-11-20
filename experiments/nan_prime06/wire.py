"""Small closed local transport contract; no model, labels or remote clients."""
import hashlib
import json


def prompt_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def validate_request(body, prompts, backend, cfg):
    allowed={'model','messages','stream','stream_options','temperature','top_p','max_tokens','max_completion_tokens','n'}
    if set(body)-allowed:raise ValueError('Unsupported request fields: '+','.join(sorted(set(body)-allowed)))
    if body.get('model')!='nanbeige-451ed48c-'+backend:raise ValueError('Unregistered model')
    if body.get('temperature',0)!=0 or body.get('top_p',1)!=1 or body.get('n',1)!=1:raise ValueError('Greedy B1 only')
    if body.get('max_tokens',body.get('max_completion_tokens',cfg['request_max_tokens']))!=cfg['request_max_tokens']:raise ValueError('Changed global request cap')
    if body.get('stream_options',{}) not in ({},{'include_usage':True}):raise ValueError('Unsupported stream options')
    messages=body.get('messages')
    if not isinstance(messages,list) or len(messages)!=1 or set(messages[0])!={'role','content'} or messages[0]['role']!='user' or not isinstance(messages[0]['content'],str):
        raise ValueError('One exact registered user prompt required; no tools or system additions')
    key=prompt_sha(messages[0]['content'])
    if key not in prompts or prompts[key]['prompt']!=messages[0]['content']:raise ValueError('Unknown prompt')
    if type(body.get('stream',False)) is not bool:raise ValueError('Invalid stream flag')
    return key,prompts[key]


def stream_chunks(identifier,model,text,usage,finish):
    common={'id':identifier,'object':'chat.completion.chunk','created':0,'model':model}
    yield {**common,'choices':[{'index':0,'delta':{'role':'assistant','content':text},'finish_reason':None}]}
    yield {**common,'choices':[{'index':0,'delta':{},'finish_reason':finish}]}
    yield {**common,'choices':[],'usage':usage}


def sse_bytes(chunks):
    return ''.join('data: '+json.dumps(c)+'\n\n' for c in chunks).encode('utf-8')+b'data: [DONE]\n\n'
