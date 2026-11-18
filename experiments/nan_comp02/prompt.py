"""Public-only prompts fixed before any new checkpoint outcomes."""
import json
from pathlib import Path

PUBLIC_FIELDS={'case_id','root','split','family','inputs','outputs','query','public_sha256'}
HEADER=("A series is a list of coefficients of x^0, x^1, x^2, x^3, x^4.\n"
        "The SAME SINGLE operator transforms both inputs into their outputs.\n"
        "I(f)=f; N(f)=f(-x); S(f)=f(x^2); R(f)=1/f.\n"
        "All lists are truncated after x^4. Identify the operator.\n"
        "Reply with exactly one letter: I, N, S, or R. No explanation.\n")
TUTORIAL=("\nWorked examples for the input [1, 1, 0, 0, 0]:\n"
          "[1, 1, 0, 0, 0] -> [1, 1, 0, 0, 0] : I\n"
          "[1, 1, 0, 0, 0] -> [1, -1, 0, 0, 0] : N\n"
          "[1, 1, 0, 0, 0] -> [1, 0, 1, 0, 0] : S\n"
          "[1, 1, 0, 0, 0] -> [1, -1, 1, -1, 1] : R\n")

def body(row,variant):
    if set(row)!=PUBLIC_FIELDS:raise ValueError('Nonpublic or missing fields in model record')
    if variant not in ['direct','worked','erased']:raise ValueError(variant)
    text=HEADER+(TUTORIAL if variant!='direct' else '')+'\nIdentify the single operator in this new task:\n'
    for i,(x,y) in enumerate(zip(row['inputs'],row['outputs']),1):
        output='[redacted]' if variant=='erased' else str(y)
        text+=f'Input {i}: {x}\nOutput {i}: {output}\n'
    return text+'Operator?'

def tokens(tok,snapshot,row,variant):
    t=tok.tok;cfg=json.loads((Path(snapshot)/'tokenizer_config.json').read_text(encoding='utf-8'))
    t.chat_template=cfg['chat_template'];t.bos_token=cfg['bos_token'];t.eos_token=cfg['eos_token']
    text=t.apply_chat_template([{'role':'user','content':body(row,variant)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return t.encode(text+'\nOperator:',add_special_tokens=False)

def aliases(tok):
    result={}
    for op in 'INSR':
        ids=[];t=tok.tok
        for text in [op,' '+op]:
            values=t.encode(text,add_special_tokens=False)
            if len(values)!=1:raise ValueError(f'Answer form not one token: {text!r}')
            ids.append(int(values[0]))
        # SentencePiece/ByteLevel normalization can prepend a space even when
        # encoding a bare character. Include its raw vocab piece only after
        # verifying the exact one-token decoded spelling.
        raw=t.convert_tokens_to_ids(op)
        if raw is not None:ids.append(int(raw))
        result[op]=sorted({i for i in ids if t.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False) in [op,' '+op]})
        if not result[op]:raise ValueError(f'No round-tripping answer token for {op}')
    flat=[i for ids in result.values() for i in ids]
    if len(flat)!=len(set(flat)):raise ValueError('Ambiguous operator aliases')
    return result
