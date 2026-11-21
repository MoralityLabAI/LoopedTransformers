"""Anchored grid/plan/finite-label grammars, no scanning or answer forcing."""
import json
import re

TICK = chr(96)

def parse(text, schema, *, final=False):
    def done(status, value=None, reason='outside_grammar'):
        return {'status':status, 'value':value, 'reason':reason}
    def pending():
        return done('rejected' if final else 'pending', reason='incomplete' if final else 'prefix')
    if len(text)>schema.get('max_characters',2048):
        return done('rejected',reason='character_limit')
    s=text.strip()
    # A single complete wrapper is accepted. A fence must be closed: an open
    # fence around an otherwise complete JSON object remains pending.
    wrapped=False
    if s.startswith(TICK*3):
        m=re.match('^'+TICK*3+r'(?:json)?\s*\n',s)
        if not m:
            return pending() if any(v.startswith(s) for v in (TICK*3+'\n',TICK*3+'json\n')) else done('rejected')
        body=s[m.end():]
        if body.endswith(TICK*3):
            s=body[:-3].strip();wrapped=True
        else:
            # The closing fence may arrive over several tokens.
            body=body.rstrip(TICK).rstrip()
            q=parse(body,schema,final=False)
            return pending() if q['status'] in ('pending','accepted') else q
    elif s.startswith('**') or s.startswith(TICK):
        marker='**' if s.startswith('**') else TICK
        body=s[len(marker):]
        if body.endswith(marker) and len(body)>len(marker):
            s=body[:-len(marker)].strip();wrapped=True
        else:
            q=parse(body.rstrip('*'+TICK),schema,final=False)
            return pending() if q['status'] in ('pending','accepted') else q
    if schema['type']=='label':
        forms={x:x for x in schema['allowed']}
        forms.update({x+'.':x for x in schema['allowed']})
        if s in forms:
            return done('accepted',forms[s],'complete')
        return pending() if any(x.startswith(s) for x in forms) else done('rejected')
    if not s:
        return pending()
    if s[0]!='[':
        return done('rejected')
    if schema['type']=='grid' and re.search(r'[^\[\],0-9\s]',s):
        return done('rejected')
    # For plans, prohibit arbitrary text outside the finite action alphabet.
    if schema['type']=='plan' and re.search(r'[^\[\],"a-z_\s]',s):
        return done('rejected')
    try:
        value=json.loads(s)
    except json.JSONDecodeError:
        # Lexically plausible prefixes stay pending until the fixed budget;
        # this is not a search over repairs or candidate answers.
        return pending()
    if schema['type']=='grid':
        valid=(isinstance(value,list) and 1<=len(value)<=12 and
               all(isinstance(r,list) and 1<=len(r)<=12 for r in value) and
               len({len(r) for r in value})==1 and
               all(type(v) is int and 0<=v<=9 for r in value for v in r))
    elif schema['type']=='plan':
        valid=(isinstance(value,list) and 1<=len(value)<=6 and
               all(type(v) is str and v in schema['allowed'] for v in value))
    else:
        raise ValueError('Unknown public schema')
    return done('accepted',value,'complete') if valid else done('rejected')
