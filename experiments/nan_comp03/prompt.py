"""Fixed reciprocal curriculum and matched public-only controls."""
import json
from pathlib import Path
from experiments.nan_comp02.prompt import PUBLIC_FIELDS,aliases

HEADER=("A series lists coefficients of x^0 through x^4.\n"
        "The SAME SINGLE operator transforms both inputs into their outputs.\n"
        "I(f)=f; N(f)=f(-x); S(f)=f(x^2); R(f)=1/f.\n"
        "Truncate every output after x^4. Identify the operator.\n"
        "Reply with one letter I, N, S, or R. No explanation.\n")
BALANCED=("\nExamples for input [1, 1, 0, 0, 0]:\n"
          "I -> [1, 1, 0, 0, 0]\nN -> [1, -1, 0, 0, 0]\n"
          "S -> [1, 0, 1, 0, 0]\nR -> [1, -1, 1, -1, 1]\n")
RECIPROCAL_EXAMPLES=[
    ([1,1,1,0,0],[1,-1,0,1,-1]),
    ([1,-1,0,1,0],[1,1,1,0,-1]),
    ([1,1,-1,0,0],[1,-1,2,-3,5]),
]
RULE=("\nR means the SERIES reciprocal, not reciprocals of individual coefficients.\n"
      "If f=[1,a1,a2,a3,a4] and R(f)=[1,b1,b2,b3,b4], then\n"
      "b1=-a1; b2=-(a1*b1+a2); b3=-(a1*b2+a2*b1+a3);\n"
      "b4=-(a1*b3+a2*b2+a3*b1+a4).\n"
      "The product f*R(f) has coefficients [1,0,0,0,0].\n"
      "For [1,1,-1,0,0]: b1=-1, b2=2, b3=-3, b4=5.\n"
      +''.join(f"{x} -> {y} : R\n" for x,y in RECIPROCAL_EXAMPLES))

def body(row,variant):
    if set(row)!=PUBLIC_FIELDS:raise ValueError('Nonpublic fields')
    if variant not in ('balanced','curriculum','erased'):raise ValueError(variant)
    text=HEADER+BALANCED+(RULE if variant!='balanced' else '')+'\nNew task:\n'
    for i,(x,y) in enumerate(zip(row['inputs'],row['outputs']),1):
        output='[redacted]' if variant=='erased' else str(y)
        text+=f'Input {i}: {x}\nOutput {i}: {output}\n'
    return text+'Operator?'

def tokens(tok,snapshot,row,variant):
    t=tok.tok;cfg=json.loads((Path(snapshot)/'tokenizer_config.json').read_text(encoding='utf-8'))
    t.chat_template=cfg['chat_template'];t.bos_token=cfg['bos_token'];t.eos_token=cfg['eos_token']
    text=t.apply_chat_template([{'role':'user','content':body(row,variant)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return t.encode(text+'\nOperator:',add_special_tokens=False)
