"""Independent transcript and collision-bound audit; standard library only."""
import sys,json,hashlib,math
from pathlib import Path
p=Path(sys.argv[1]); r=json.loads(p.read_text()); k=r['k']; n=1<<k
assert k==8 and r['bit_cap']==4
assert hashlib.sha256(Path(__file__).with_name('pilot.py').read_bytes()).hexdigest()==r['source_sha256']
for s in range(1,k+1):
    N=1<<s
    actual=max(sum(max(s-x.bit_count(),s-(x^((1<<d)-1)).bit_count()) for x in range(N)) for d in range(s+1))
    expected=s*N//2+sum(math.comb(s,i)*abs(s-2*i) for i in range(s+1))//2
    assert actual==expected
seen=set(); cases=0; full_rows=0
for m in r['results']:
    root,arm=m['root'],m['arm']; assert (root,arm) not in seen; seen.add((root,arm))
    perm=m['permutation']; assert sorted(perm)==list(range(k))
    assert set(m['train_ids']).isdisjoint(m['heldout_ids']); assert sorted(m['train_ids']+m['heldout_ids'])==list(range(n))
    full=m['full']; mapping=full['query_map']; hist=[mapping.count(c) for c in range(k)]; assert hist==full['histogram']
    assert full['query_codes']==sum(s>0 for s in hist)
    ceiling=sum(s/2+sum(math.comb(s,i)*abs(s-2*i) for i in range(s+1))/2**(s+1) for s in hist)/k
    assert abs(ceiling-full['collision_ceiling'])<1e-12 and full['correct']<=ceiling*n*k
    dec={}; alice={}
    for split,ids in [('train',m['train_ids']),('heldout',m['heldout_ids']),('full',list(range(n)))]:
        q=m[split]; assert q['total']==len(ids)*k==len(q['correct_bits'])==len(q['transcripts']); assert q['correct']==q['correct_bits'].count('1'); assert q['exact_rows']==sum(q['correct_bits'][i:i+k]=='1'*k for i in range(0,q['total'],k))
        for i,x in enumerate(ids):
            for j in range(k):
                idx=i*k+j; t=q['transcripts'][idx]; assert isinstance(t,int) and 0<=t<16 and t//2==mapping[j]
                a=(x,t//2); reply=t%2; assert alice.setdefault(a,reply)==reply
                label=(x>>perm[j])&1; pred=label if q['correct_bits'][idx]=='1' else 1-label
                assert dec.setdefault((j,t),pred)==pred
                assert full['transcripts'][x*k+j]==t and full['correct_bits'][x*k+j]==q['correct_bits'][idx]
                cases+=1
    full_rows+=n
print(json.dumps(dict(models=len(seen),scored_transcript_rows=cases,distinct_full_cases=full_rows*k,source_sha256=r['source_sha256'],receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),collision_group_sizes_checked=list(range(1,k+1)),status='PASS'),indent=2))
