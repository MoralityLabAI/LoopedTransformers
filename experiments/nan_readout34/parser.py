"""Bounded whole-reply model-side stop grammar. No gold access or answer scan."""
import re


def parse(text,schema):
    if not isinstance(text,str) or len(text)>512 or not text.isascii():return None
    text=text.strip(' \t\r\n');kind=schema['readout']
    if kind=='final':
        m=re.fullmatch(r'FINAL=(R[0-7]);END',text)
        return dict(final_room=m[1]) if m else None
    if kind=='single_checkpoint':
        m=re.fullmatch(r'AT=(R[0-7]);END',text)
        return dict(single_room=m[1]) if m else None
    if kind=='checkpoint_array':
        m=re.fullmatch(r'CHECKS=(R[0-7]),(R[0-7]),(R[0-7]);END',text)
        return dict(checkpoint_rooms=list(m.groups())) if m else None
    if kind!='packet':raise ValueError('Unknown readout')
    times=schema['checkpoints']
    pattern=';'.join(f'T{t}=(R[0-7])' for t in times)+r';PATH=([A-Z0-9>]+);FINAL=(R[0-7]);END'
    m=re.fullmatch(pattern,text)
    if not m:return None
    path=m[4].split('>')
    if not 2<=len(path)<=16 or any(n not in schema['nodes'] for n in path):return None
    return dict(checkpoint_rooms=list(m.groups()[:3]),path=path,final_room=m[5])
