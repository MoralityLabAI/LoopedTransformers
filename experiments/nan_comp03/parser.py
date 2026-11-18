"""Frozen prefix-closed stopping grammar; never search prose for an answer."""
import re

MAX_CHARACTERS=64
FORMS={}
for op in 'INSR':
    for value in (op,'**'+op+'**','*'+op+'*','`'+op+'`'):
        FORMS[value]=op
        FORMS[value+'.']=op

def parse(text,*,final=False):
    if len(text)>MAX_CHARACTERS:
        return {'status':'rejected','operator':None,'reason':'character_limit'}
    value=re.sub(r'\s+','',text)
    if value in FORMS:
        return {'status':'accepted','operator':FORMS[value],'reason':'complete_grammar'}
    if any(s.startswith(value) for s in FORMS):
        return {'status':'rejected' if final else 'pending','operator':None,'reason':'incomplete_at_limit' if final else 'grammar_prefix'}
    return {'status':'rejected','operator':None,'reason':'outside_grammar'}
