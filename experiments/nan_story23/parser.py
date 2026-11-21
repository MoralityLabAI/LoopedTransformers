"""Anchored bounded trace packets; no substring, prose or answer rescue."""
import re


def parse(text,schema):
    if not isinstance(text,str) or len(text)>512:
        return None
    pieces=[]
    for i,time in enumerate(schema['checkpoints']):
        pieces.append(r'\s*T0*'+str(time)+r'\s*=\s*(?P<r'+str(i)+r'>R[0-7])\s*;')
    node=r'(?:E[0-9]{2}|P[0-3]|R[0-7])'
    pieces.append(r'\s*PATH\s*=\s*(?P<path>'+node+r'(?:\s*>\s*'+node+r'){1,15})\s*;')
    pieces.append(r'\s*FINAL\s*=\s*(?P<final>R[0-7])\s*;\s*END\s*')
    matched=re.fullmatch(''.join(pieces),text)
    if matched is None:
        return None
    path=[v.strip() for v in matched['path'].split('>')]
    if any(v not in schema['nodes'] for v in path):
        return None
    return dict(intermediate_rooms=[matched['r'+str(i)] for i in range(3)],path=path,final_room=matched['final'])
