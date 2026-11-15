"""RAM-12 reproduction adapter for the original unmodified RAM-08 model.

Production model and tokenizer code are reused directly from the existing
experiments/lean_ram08/model.py. The hashes below pin original training bytes.
"""
from __future__ import annotations
from pathlib import Path
import sys
ORIGINAL=Path(__file__).resolve().parents[1]/'lean_ram08'
sys.path.insert(0,str(ORIGINAL))
from model import Config, Policy, encode

ARCHIVE_SHA='a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc'
DATA_SHA='f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f'
EXPECTED={
    'loop32':(29,'4a88bc80456743a89a70882911060cc90e76785a5ae1d435a0e46a8f8c09778f'),
    'loop64':(43,'5cbd327904714ac698897e1c2333cae9ec4125aeb70e7c56cd6236db8a737dac'),
    'flat64':(29,'2aa101edbd56f6563ebb2edcfc8c61c07dcc027559ea4cdc551187d6ec5303d8'),
}
