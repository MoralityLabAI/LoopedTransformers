"""Source-only prompt factorization. Gold assistance is explicit and arm-local."""
import json
from pathlib import Path

OPS='INSR'
RULE=('For f=[1,a1,a2,a3,a4] and its inverse [1,b1,b2,b3,b4], '
      'b1=-a1; b2=-(a1*b1+a2); b3=-(a1*b2+a2*b1+a3); '
      'b4=-(a1*b3+a2*b2+a3*b1+a4).\n')
NUMBER_REPLY='Reply with only the signed integer in square brackets, such as [7]. No explanation.'
DEFS='I(f)=f; N(f)=f(-x); S(f)=f(x^2); R(f)=1/f. All coefficient lists stop at x^4.\n'

def make_prompts(inputs,refs,worlds,codes,order):
    prompts={}
    def add(name,text,kind):prompts[name]={'text':text,'kind':kind}
    focus=inputs[0]
    for n in range(1,5):
        stem=f'The formal power series f has coefficients {focus} from x^0 through x^4. Its constant coefficient is 1. Find the coefficient of x^{n} in the multiplicative inverse 1/f (not term-wise reciprocals).\n'
        add(f'free/b{n}',stem+NUMBER_REPLY,'integer')
        add(f'scaffold/b{n}',stem+RULE+NUMBER_REPLY,'integer')
        add(f'copy/b{n}',f'A verified inverse coefficient of x^{n} is {refs["R"][0][n]}. Copy that integer.\n'+NUMBER_REPLY,'integer')
    for index,op in enumerate(worlds):
        observations=''.join(f'Input {i+1}: {x}\nOutput {i+1}: {y}\n' for i,(x,y) in enumerate(zip(inputs,refs[op])))
        add(f'select/{index}','Identify the SAME SINGLE operator from these two examples.\n'+DEFS+observations+'Reply only I, N, S, or R. No explanation.','operator')
        for name,labels,kind in (('table',{g:g for g in OPS},'operator'),('coded',codes,'code')):
            table=''.join(f'Entry {labels[g]}: output 1 = {refs[g][0]}; output 2 = {refs[g][1]}\n' for g in order)
            text='Match BOTH observed outputs to ONE supplied reference entry. No calculation is needed.\n'+observations+'Reference entries for the same inputs:\n'+table+'Reply only the matching entry label. No explanation.'
            add(f'{name}/{index}',text,kind)
        add(f'labelcopy/{index}',f'The verified operator label is {op}. Copy only that letter.','operator')
        add(f'codecopy/{index}',f'The verified reference code is {codes[op]}. Copy only that letter.','code')
    blind='Identify the SAME SINGLE operator from these two examples.\n'+DEFS+''.join(f'Input {i+1}: {x}\nOutput {i+1}: [redacted]\n' for i,x in enumerate(inputs))+'Reply only I, N, S, or R. No explanation.'
    add('erased',blind,'operator');add('erased_repeat',blind,'operator')
    assert len(prompts)==24
    return prompts

def tokens(tok,snapshot,prompt):
    t=tok.tok;cfg=json.loads((Path(snapshot)/'tokenizer_config.json').read_text(encoding='utf-8'))
    t.chat_template=cfg['chat_template'];t.bos_token=cfg['bos_token'];t.eos_token=cfg['eos_token']
    text=t.apply_chat_template([{'role':'user','content':prompt['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    suffix='\nAnswer:' if prompt['kind']=='integer' else ('\nCode:' if prompt['kind']=='code' else '\nOperator:')
    return t.encode(text+suffix,add_special_tokens=False)
