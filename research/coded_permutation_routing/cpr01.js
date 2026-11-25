// CPR-01 deterministic finite oracle and recurrent learned comparator pilot.
function cpr01() {
  "use strict";
  const assert=(ok,msg)=>{if(!ok)throw Error(msg)};
  const eq=(a,b)=>a.length===b.length&&a.every((x,i)=>x===b[i]);
  const range=n=>Array.from({length:n},(_,i)=>i);
  const inv=p=>p.reduce((s,x,i)=>s+p.slice(i+1).filter(y=>x>y).length,0);
  const key=p=>p.join(",");
  function matchings(n,i=0) {
    if(i>=n-1)return [[]];
    return [...matchings(n,i+1),...matchings(n,i+2).map(m=>[i,...m])];
  }
  function step(p,m) {
    const q=p.slice();
    for(const i of m)[q[i],q[i+1]]=[q[i+1],q[i]];
    return q;
  }
  function oracle(n) {
    const start=range(n), moves=matchings(n).filter(m=>m.length);
    const d=new Map([[key(start),0]]),queue=[start];
    for(let j=0;j<queue.length;j++){
      const p=queue[j],next=d.get(key(p))+1;
      for(const m of moves){const q=step(p,m),k=key(q);
        if(!d.has(k)){d.set(k,next);queue.push(q)}
      }
    }
    return {d,queue};
  }
  function route(p,mode,decide=(a,b)=>a>b,width=16) {
    const n=p.length, original=p.slice();
    let dest=p.slice(), payload=p.map((_,i)=>(Math.imul(i+1,40503)^0xa5a5)&((1<<width)-1)),initial=payload.slice();
    let rounds=0,swaps=0,decisions=0, falseSwap=0,miss=0;
    while(!eq(dest,range(n))&&rounds<n*2){
      const m=[];
      for(let i=mode==="odd-even"?rounds%2:0;i<n-1;){
        decisions++;
        const take=decide(dest[i],dest[i+1]);
        if(take&&dest[i]<dest[i+1])falseSwap++;
        if(!take&&dest[i]>dest[i+1])miss++;
        if(take){m.push(i);i+=2}
        else i+=mode==="odd-even"?2:1;
      }
      dest=step(dest,m);payload=step(payload,m);swaps+=m.length;rounds++;
      if(!m.length&&mode!=="odd-even")break;
    }
    const target=range(n).map(i=>initial[original.indexOf(i)]);
    const success=eq(dest,range(n))&&eq(payload,target);
    return {success,rounds,swaps,payloadBits:2*width*swaps,tagBits:2*Math.ceil(Math.log2(n))*swaps,decisions,falseSwap,miss};
  }
  const exact=[];
  for(let n=2;n<=7;n++){
    const {d,queue}=oracle(n),stat={n,permutations:queue.length,optimalRounds:0,oddEvenRounds:0,greedyRounds:0,oddEvenOptimal:0,greedyOptimal:0,oddEvenMaxGap:0,greedyMaxGap:0,totalInversions:0,totalPayloadBits:0,totalTagBits:0};
    assert(queue.length===range(n).reduce((x,i)=>x*(i+1),1),"coverage");
    for(const p of queue){
      const o=d.get(key(p)), a=route(p,"odd-even"),b=route(p,"greedy"),v=inv(p);
      assert(a.success&&b.success,"route reconstruction");
      assert(a.swaps===v&&b.swaps===v,"inversion lower bound");
      assert(a.rounds>=o&&b.rounds>=o,"oracle lower bound");
      stat.optimalRounds+=o;stat.oddEvenRounds+=a.rounds;stat.greedyRounds+=b.rounds;
      stat.oddEvenOptimal+=+(a.rounds===o);stat.greedyOptimal+=+(b.rounds===o);
      stat.oddEvenMaxGap=Math.max(stat.oddEvenMaxGap,a.rounds-o);
      stat.greedyMaxGap=Math.max(stat.greedyMaxGap,b.rounds-o);
      stat.totalInversions+=v;stat.totalPayloadBits+=a.payloadBits;stat.totalTagBits+=a.tagBits;
    }
    exact.push(stat);
  }
  let stable=0;
  for(let v=0;v<243;v++){
    let q=v;const keys=range(5).map(()=>{const x=q%3;q=Math.floor(q/3);return x});
    const order=range(5).sort((a,b)=>keys[a]-keys[b]||a-b);
    const p=range(5).map(i=>order.indexOf(i));
    assert(route(p,"odd-even").success,"stable rank route");stable++;
  }
  const relay=[];
  for(let width=1;width<=6;width++){
    let pairs=0,badSideInfo=0;
    for(let a=0;a<2**width;a++)for(let b=0;b<2**width;b++){
      const c=a^b;
      assert((c^a)===b&&(c^b)===a,"xor relay");
      badSideInfo+=+((c^(a^1))!==b);pairs++;
    }
    relay.push({width,pairs,badSideInfo,uncodedBroadcastBits:2*width,codedBroadcastBits:width,uncodedTotalTxBits:4*width,codedTotalTxBits:3*width,uncodedUnicastTotalBits:4*width,codedUnicastTotalBits:4*width});
  }
  function rng(seed){let x=seed>>>0;return ()=>{x=(Math.imul(x,1664525)+1013904223)>>>0;return x/4294967296}}
  function diff(a,b){return range(6).map(i=>((a>>>i)&1)-((b>>>i)&1))}
  function fit(seed){
    const rand=rng(seed),w=range(6).map(()=>0);
    for(let t=0;t<24000;t++){
      const a=Math.floor(rand()*16),b=Math.floor(rand()*16);
      if(a===b)continue;
      const x=diff(a,b),y=+(a>b),z=x.reduce((s,v,i)=>s+v*w[i],0);
      const g=(y-1/(1+Math.exp(-z)))*0.15;
      for(let i=0;i<6;i++)w[i]+=g*x[i];
    }
    return w;
  }
  function perm(n,rand){const p=range(n);for(let i=n-1;i>0;i--){const j=Math.floor(rand()*(i+1));[p[i],p[j]]=[p[j],p[i]]}return p}
  const neural=[],raw=[];
  for(const seed of [17,31,53]){
    const w=fit(seed),decide=(a,b)=>diff(a,b).reduce((s,v,i)=>s+v*w[i],0)>0;
    for(const n of [8,16,32,64]){
      const rand=rng(100000+seed*100+n),st={seed,n,roots:64,successes:0,oracleSuccesses:0,falseSwaps:0,misses:0,totalRounds:0,totalSwaps:0,oracleRounds:0};
      for(let i=0;i<64;i++){
        const p=perm(n,rand),a=route(p,"odd-even",decide),b=route(p,"odd-even");
        assert(b.success,"symbolic ceiling");
        st.successes+=+a.success;st.oracleSuccesses+=+b.success;st.falseSwaps+=a.falseSwap;st.misses+=a.miss;
        st.totalRounds+=a.rounds;st.totalSwaps+=a.swaps;st.oracleRounds+=b.rounds;
        raw.push({seed,n,root:i,permutation:p,learned:a,exact:b});
      }
      neural.push(st);
    }
  }
  return {summary:{schema:1,exact,stableKeyFixtures:stable,relay,neural,evidence:"exact finite line and relay; exploratory learned comparator only"},raw};
}
const result=cpr01();
const fs=require("node:fs");
if(process.argv[2]) fs.writeFileSync(process.argv[2],JSON.stringify(result.summary,null,2)+"\n");
else console.log(JSON.stringify(result.summary,null,2));
if(process.argv[3]) fs.writeFileSync(process.argv[3],JSON.stringify(result.raw,null,2)+"\n");
