"""Anchored, prefix-closed two-task grammar; never search prose for answers."""
from __future__ import annotations
import re

from experiments.nan_comp03.parser import parse as class_parse

NUMERAL_FORMS={}
for n in range(-999,1000):
    representations=[str(n)]
    if n>=0:
        representations.append("+"+str(n))
    for value in representations:
        for rendered in ("["+value+"]","**["+value+"]**"):
            NUMERAL_FORMS[rendered]=n


def number_parse(text:str,*,final:bool=False)->dict:
    if len(text)>64:
        return {"status":"rejected","value":None,"reason":"character_limit"}
    value=re.sub(r"\s+","",text)
    if value in NUMERAL_FORMS:
        return {"status":"accepted","value":NUMERAL_FORMS[value],"reason":"complete_grammar"}
    if any(form.startswith(value) for form in NUMERAL_FORMS):
        return {"status":"rejected" if final else "pending","value":None,
                "reason":"incomplete_at_limit" if final else "grammar_prefix"}
    return {"status":"rejected","value":None,"reason":"outside_grammar"}


def parse(text:str,kind:str,*,final:bool=False)->dict:
    if kind=="classification":
        return class_parse(text,final=final)
    if kind=="coefficient":
        return number_parse(text,final=final)
    raise ValueError("unknown NAN-COMP-04 task")
