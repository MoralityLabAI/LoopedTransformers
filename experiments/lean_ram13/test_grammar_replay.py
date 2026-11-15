"""Independently compare RAM13 source-visible grammar to all RAM08 old-source observations."""
import argparse,collections,hashlib,json,zipfile
from pathlib import Path
import grammar

p=argparse.ArgumentParser();p.add_argument('--ram08',type=Path,required=True);args=p.parse_args()
archive=args.ram08.read_bytes()
assert hashlib.sha256(archive).hexdigest()=='a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc'
with zipfile.ZipFile(args.ram08) as z:
 data=z.read('data/transitions.jsonl')
 assert hashlib.sha256(data).hexdigest()=='f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f'
 rows=[json.loads(x) for x in data.decode().splitlines() if x.strip()]
assert len(rows)==20930
groups=collections.OrderedDict()
for row in rows:groups.setdefault((row['source_file'],row['theorem'],tuple(row['prefix'])),[]).append(row)
assert len(groups)==714
for root,actual in groups.items():
 expected=[(x['family'],x['script']) for x in actual]
 proposed=grammar.candidates(actual[0]['before'])
 if proposed!=expected:raise AssertionError(('candidate grammar mismatch',root,proposed,expected))
print('RAM13_REGISTERED_GRAMMAR_EXACT_MATCH',len(groups),'states',len(rows),'compiler observations')
