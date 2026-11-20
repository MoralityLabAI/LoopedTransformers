"""Bounded unforced output grammars; integer digits need an explicit delimiter."""
import re

MAX_CHARACTERS=64
WRAPPED={}
NUMBERS={str(n):n for n in range(-999,1000)}
NUMBERS.update({'+'+str(n):n for n in range(1000)})
NUMBERS['-0']=0
for text,value in NUMBERS.items():
    for left,right in (('[',']'),('**','**'),('*','*'),('`','`'),('**[',']**'),('`[',']`')):
        WRAPPED[left+text+right]=value
PREFIXES={word[:i] for word in list(WRAPPED)+list(NUMBERS) for i in range(len(word)+1)}

def parse(text,kind,*,eos=False,limit=False):
    if len(text)>MAX_CHARACTERS:return {'status':'rejected','value':None,'reason':'character_limit'}
    if kind in ('operator','code'):
        alphabet='INSR' if kind=='operator' else 'ABCD';forms={}
        for op in alphabet:
            for s in (op,'**'+op+'**','*'+op+'*','`'+op+'`'):
                forms[s]=op;forms[s+'.']=op
        s=re.sub(r'\s+','',text)
        if s in forms:return {'status':'accepted','value':forms[s],'reason':'complete_grammar'}
        pending=any(v.startswith(s) for v in forms)
    elif kind=='integer':
        # Never merge distinct digit groups, e.g. "1 7" into 17.
        if re.search(r'[0-9]\s+[0-9]',text):return {'status':'rejected','value':None,'reason':'split_digits'}
        s=re.sub(r'\s+','',text.replace('\u2212','-'))
        terminated=bool(text and text[-1].isspace()) or s.endswith('.') or eos
        if s.endswith('.'):
            s=s[:-1]
            if s not in WRAPPED and s not in NUMBERS:return {'status':'rejected','value':None,'reason':'outside_grammar'}
        if s in WRAPPED:return {'status':'accepted','value':WRAPPED[s],'reason':'complete_grammar'}
        if s in NUMBERS and terminated:return {'status':'accepted','value':NUMBERS[s],'reason':'terminated_integer'}
        pending=s in PREFIXES
    else:raise ValueError(kind)
    return {'status':'pending' if pending and not (eos or limit) else 'rejected','value':None,
            'reason':'grammar_prefix' if pending and not (eos or limit) else ('incomplete_at_limit' if pending else 'outside_grammar')}
