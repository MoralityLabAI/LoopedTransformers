"""Independent absolute-mask table arithmetic and chronological frontier reference."""
import json,hashlib,math,itertools as it
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction as F
H=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def par(a):return a.bit_count()%2
def submasks(a):
    x=a
    while True:
        yield x
        if not x:return
        x=(x-1)&a

def group(rows,obs,p):
    u,v=F(str(p)).as_integer_ratio();v-=u;g={}
    for a,y in zip(rows,obs):
        x,z=g.get(a,(1,1));g[a]=(x*(u if y else v),z*(v if y else u))
    return g

def independent(g,active):
    fixed=g.get(0,(1,1))[0];lo=hi=fixed;piv={};rank=0
    for a,(x,y) in sorted(((a,p) for a,p in g.items() if a),key=lambda t:(F(min(t[1]),max(t[1])),t[0])):
        r=a
        while r and r&-r in piv:r^=piv[r&-r]
        if r:piv[r&-r]=r;lo*=x+y;hi*=x+y;rank+=1
        else:lo*=min(x,y);hi*=max(x,y)
    return lo*2**(active.bit_count()-rank),hi*2**(active.bit_count()-rank)

def component(g,active):
    todo=set(1<<i for i in range(active.bit_length()) if active>>i&1);lo=hi=g.get(0,(1,1))[0]
    while todo:
        s=min(todo);old=0
        while old!=s:
            old=s
            for a in g:
                if a&s:s|=a
        todo={a for a in todo if not a&s};f={a:p for a,p in g.items() if a&s}
        # Main worlds have an observed unary for every active variable after conditioning.
        assert all(1<<i in f for i in range(s.bit_length()) if s>>i&1)
        if s.bit_count()<=7:l=u=sum(math.prod(p[par(a&h)] for a,p in f.items()) for h in submasks(s))
        else:l,u=independent(f,s)
        lo*=l;hi*=u
    return lo,hi

def mb(g,active,cap,trace):
    # Absolute assignment masks, unlike pilot's packed scope indices.
    fac=[(a,{h:(v[par(h)],)*2 for h in submasks(a)}) for a,v in sorted(g.items())];used=0
    for a in g:used|=a
    for i in range(active.bit_length()):
        a=1<<i
        if a&(active&~used):fac.append((a,{0:(1,1),a:(1,1)}))
    for v,scopes,anchor in trace:
        def union(a):
            s=0
            for k,_ in fac:
                if k&a:s|=k
            return s
        assert active>>v&1;bit=1<<v;assert bit==min((1<<i for i in range(active.bit_length()) if active>>i&1),key=lambda a:(union(a).bit_count(),a))
        active^=bit;bucket=[f for f in fac if f[0]&bit];fac=[f for f in fac if not f[0]&bit];bins=[];masks=[]
        def scope_key(a):return tuple(i for i in range(a.bit_length()) if a>>i&1)
        for a,t in sorted(bucket,key=lambda f:(-f[0].bit_count(),scope_key(f[0]))):
            ok=[j for j,s in enumerate(masks) if (s|a).bit_count()<=cap]
            if not ok:bins.append([(a,t)]);masks.append(a)
            else:j=min(ok,key=lambda j:((masks[j]|a).bit_count(),j));bins[j].append((a,t));masks[j]|=a
        assert scopes==[a.bit_count() for a in masks] and all(x<=cap for x in scopes)
        assert anchor==max(range(len(bins)),key=lambda j:(scopes[j],len(bins[j]),-j))
        for j,(a,fs) in enumerate(zip(masks,bins)):
            out=a^bit;table={}
            for h in submasks(out):
                lo0,hi0=(math.prod(t[h&s][i] for s,t in fs) for i in (0,1));lo1,hi1=(math.prod(t[(h|bit)&s][i] for s,t in fs) for i in (0,1))
                table[h]=(lo0+lo1,hi0+hi1) if j==anchor else (min(lo0,lo1),max(hi0,hi1))
            fac.append((out,table))
    assert active==0 and all(a==0 for a,t in fac);return tuple(math.prod(t[0][i] for a,t in fac) for i in (0,1))

def frontier_exact(g,n,event=None):
    # Process original coordinates in order. Keep assignments only if future factors use them.
    factors=list(g.items())+([] if event is None else [(event,(0,1))]);dp={0:1};peak=1
    fixed=math.prod(p[0] for a,p in factors if not a);factors=[(a,p) for a,p in factors if a]
    for j in range(n):
        b=1<<j;done=[(a,p) for a,p in factors if a.bit_length()==j+1];future=0
        for a,p in factors:
            if a.bit_length()>j+1:future|=a
        future&=(2*b-1);out=defaultdict(int)
        for h,w in dp.items():
            for x in (h,h|b):
                v=w*math.prod(p[par(a&x)] for a,p in done)
                out[x&future]+=v
        dp=out;peak=max(peak,len(dp))
    return fixed*dp[0],peak

def run():
    r=json.loads((H/'receipt.json').read_text());assert r['source_sha256']==sha(H/'pilot16.py') and r['protocol_sha256']==sha(H/'PROTOCOL.md');assert r['fixture_checks']==1080
    ws={w['seed']:w for w in r['worlds']};assert len(ws)==36;refs={};checks=Counter();seen=set();max_frontier=0
    for a in r['results']:
        w=ws[a['seed']];n=w['n'];p=a['noise'];arm=a['arm'];key=(w['seed'],p,arm);assert key not in seen;seen.add(key);active=(1<<n)-1
        obs=[par(q&w['hidden'])^int(u<p) for q,u in zip(w['rows'],w['uniforms'])];assert a['obs']==obs;g=group(w['rows'],obs,p)
        if n==12 or w['family']!='dense':
            k=(w['seed'],p)
            if k not in refs:
                z,peak=frontier_exact(g,n);zs=[];max_frontier=max(max_frontier,peak)
                for q in w['queries']:
                    v,peak=frontier_exact(g,n,q);max_frontier=max(max_frontier,peak);zs.append(v)
                refs[k]=(z,zs)
            z,zs=refs[k]
        else:z=zs=None
        def check(b,gg,act,exact):
            base=independent(gg,act);assert list(map(str,base))==b['baseline'];raw=base if arm=='independent' else component(gg,act) if arm=='component7' else mb(gg,act,int(arm[2:]),b['trace']);assert list(map(str,raw))==b['raw'];lo=max(base[0],raw[0]);hi=min(base[1],raw[1]);assert (str(lo),str(hi))==(b['lo'],b['hi']) and 0<lo<=hi;checks['weight_bound_replays']+=1
            if exact is not None:assert lo<=exact<=hi and raw[0]<=exact<=raw[1];checks['exact_weight_enclosures']+=1
            return lo,hi
        check(a['root'],g,active,z)
        for i,q in enumerate(w['queries']):
            s=a['queries'][i];assert s['q']==q and s['truth']==par(q&w['hidden']);b0=[];b1=[];pivot=q&-q
            for b in (0,1):
                # Independent conditioning of grouped likelihoods. Swap pair when fixed parity is 1.
                gg={}
                for aa,(x,y) in g.items():
                    bb=aa^q if aa&pivot else aa
                    if aa&pivot and b:x,y=y,x
                    u,v=gg.get(bb,(1,1));gg[bb]=(u*x,v*y)
                exact=None if z is None else zs[i] if b else z-zs[i];l,u=check(s['weights'][b],gg,active^pivot,exact);b0.append(l);b1.append(u)
            lo,hi=F(b0[1],b0[1]+b1[0]),F(b1[1],b1[1]+b0[0]);assert (str(lo),str(hi))==(s['lo'],s['hi']);cert=1 if lo>=F(9,10) else 0 if hi<=F(1,10) else None;assert cert==s['cert'];checks['intervals']+=1
            if z is not None:assert lo<=F(zs[i],z)<=hi;checks['exact_interval_enclosures']+=1
        if len(seen)%30==0:print('audit',len(seen),flush=True)
    assert len(seen)==360
    out=dict(status='PASS',**checks,worlds=len(ws),conditions_with_exact_reference=len(refs),max_frontier_states=max_frontier,fixture_checks=r['fixture_checks'],source_sha256=sha(H/'pilot16.py'),receipt_sha256=sha(H/'receipt.json'),audit_sha256=sha(__file__),scope='independent grouped integer weights, LSB independent factors, absolute-mask mini-bucket table replay, exact chronological frontier DP, intervals and outcomes; no performance replay')
    (H/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':run()
