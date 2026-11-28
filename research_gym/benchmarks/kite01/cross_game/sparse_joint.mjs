import assert from 'node:assert/strict';

const sum = a => a.reduce((s,v)=>s+v,0n);
const prod = a => a.reduce((s,v)=>s*v,1n);
const int = v => {
  assert((typeof v==='string'&&/^(0|[1-9]\d*)$/.test(v))||(Number.isSafeInteger(v)&&v>=0),'nonnegative integer required');
  return BigInt(v);
};
const key = a => a.join(',');
function gcd(a,b){while(b){[a,b]=[b,a%b]}return a}
function frac(a,b){assert(b>0n&&a>=0n);const g=gcd(a,b);return String(a/g)+'/'+String(b/g)}
function cartesian(sets,visit,at=0,tuple=[]){
  if(at===sets.length){visit([...tuple]);return}
  for(const a of sets[at]){tuple.push(a);cartesian(sets,visit,at+1,tuple);tuple.pop()}
}
function exactKeys(obj,ids){assert(obj&&typeof obj==='object');assert.deepEqual(Object.keys(obj).sort(),[...ids].sort())}
function validate(spec){
  assert(spec.schema==='kite.sparse-joint.v1');
  const p=spec.players;
  assert(Array.isArray(p)&&p.length>=2&&p.length<=16&&p.every(x=>typeof x==='string'&&x.length>0)&&new Set(p).size===p.length,'player roster');
  exactKeys(spec.rosters,p);
  const sizes=p.map(id=>{const a=spec.rosters[id];assert(Array.isArray(a)&&a.length>=1&&a.length<=64&&a.every(v=>typeof v.id==='string'&&v.id.length>0)&&new Set(a.map(v=>v.id)).size===a.length,'action roster');return a.length});
  const D=int(spec.denominator);assert(D>0n);
  assert(Array.isArray(spec.profiles)&&spec.profiles.length>=1&&spec.profiles.length<=32&&new Set(spec.profiles.map(q=>q.id)).size===spec.profiles.length);
  const profiles=spec.profiles.map(q=>{
    assert(typeof q.id==='string'&&q.id.length>0);
    exactKeys(q.weights,p);
    const w=p.map((id,i)=>{assert(Array.isArray(q.weights[id])&&q.weights[id].length===sizes[i]);const a=q.weights[id].map(int);assert(sum(a)>0n,'empty support');return a});
    const support=w.map(a=>a.flatMap((v,i)=>v>0n?[i]:[])),totals=w.map(sum),s=support.map(a=>BigInt(a.length)),base=prod(s);
    const count=base+s.reduce((z,v,i)=>z+(BigInt(sizes[i])-v)*(base/v),0n);
    return {id:q.id,w,support,totals,count};
  });
  return {players:p,sizes,D,profiles};
}
function plan(spec,limit=4096){
  assert(Number.isSafeInteger(limit)&&limit>0&&limit<=100000);
  const v=validate(spec),all=new Map(),profiles=[];
  for(const q of v.profiles){
    assert(q.count<=BigInt(limit),'single profile exceeds payoff budget before enumeration');
    const cells=new Set();
    const add=a=>{const k=key(a);cells.add(k);all.set(k,a);assert(all.size<=limit,'union exceeds payoff budget')};
    cartesian(q.support,add);
    for(let i=0;i<v.players.length;i++){
      const outside=Array.from({length:v.sizes[i]},(_,k)=>k).filter(a=>q.w[i][a]===0n);
      if(outside.length)cartesian(q.support.map((s,j)=>i===j?outside:s),add);
    }
    assert(BigInt(cells.size)===q.count,'frontier counting discrepancy');
    profiles.push({id:q.id,support_sizes:q.support.map(a=>a.length),required_cells:cells.size});
  }
  return {full_tensor_cells:String(prod(v.sizes.map(BigInt))),required_cells:all.size,profiles,cells:[...all.values()].sort((a,b)=>{for(let i=0;i<a.length;i++)if(a[i]!==b[i])return a[i]-b[i];return 0})};
}
function certify(spec,receipts,limit=4096){
  const v=validate(spec),p=plan(spec,limit),required=new Set(p.cells.map(key)),cells=new Map();
  assert(Array.isArray(receipts)&&receipts.length===required.size,'cell count mismatch');
  for(const r of receipts){
    assert(Array.isArray(r.actions)&&r.actions.length===v.players.length&&r.actions.every((a,i)=>Number.isInteger(a)&&a>=0&&a<v.sizes[i]),'invalid action tuple');
    const k=key(r.actions);assert(required.has(k)&&!cells.has(k),'unexpected or duplicate cell');
    assert(Array.isArray(r.payoffs)&&r.payoffs.length===v.players.length,'incomplete payoff vector');
    const a=r.payoffs.map(int);assert(a.every(x=>x<=v.D),'payoff outside global bound');
    assert(typeof r.post_state_hash==='string'&&/^[a-f0-9]{64}$/.test(r.post_state_hash),'invalid state digest');
    cells.set(k,a);
  }
  assert(cells.size===required.size,'missing cell');
  const lo=v.players.map((_,i)=>[...cells.values()].reduce((z,a)=>a[i]<z?a[i]:z,v.D));
  const hi=v.players.map((_,i)=>[...cells.values()].reduce((z,a)=>a[i]>z?a[i]:z,0n));
  const span=hi.map((x,i)=>x-lo[i]),allTotals=[];
  for(const q of v.profiles){
    const total=prod(q.totals),results=[];
    let globalMax=0n,localN=0n,localD=1n;
    for(let i=0;i<v.players.length;i++){
      const sums=[];
      for(let a=0;a<v.sizes[i];a++){
        let gain=0n;
        cartesian(q.support.map((s,j)=>j===i?[a]:s),tuple=>{
          let weight=1n;for(let j=0;j<tuple.length;j++)if(j!==i)weight*=q.w[j][tuple[j]];
          gain+=weight*cells.get(key(tuple))[i];
        });
        sums.push(gain);
      }
      const inc=sum(sums.map((x,a)=>x*q.w[i][a])),best=sums.reduce((b,x,a)=>x>sums[b]?a:b,0),gain=sums[best]*q.totals[i]-inc;
      assert(gain>=0n);
      if(gain>globalMax)globalMax=gain;
      const ld=(span[i]||1n)*total;
      assert(span[i]!==0n||gain===0n);
      if(gain*localD>localN*ld){localN=gain;localD=ld}
      results.push({player:v.players[i],best_action:spec.rosters[v.players[i]][best].id,
        incumbent_utility:frac(inc,total*v.D),regret_global:frac(gain,total*v.D),regret_utility_units:frac(gain,total),
        regret_local_observed_range:frac(gain,ld),profitable:gain>0n,observed_payoff_span:String(span[i]),
        action_utilities:sums.map((x,a)=>({id:spec.rosters[v.players[i]][a].id,utility:frac(x,(total/q.totals[i])*v.D)}))});
    }
    allTotals.push({id:q.id,players:results,global_max_regret:frac(globalMax,total*v.D),
      local_max_regret_observed_range:frac(localN,localD),restricted_exact_nash:globalMax===0n,
      restricted_global_le_020:5n*globalMax<=total*v.D,restricted_local_le_020:5n*localN<=localD,
      required_cells:Number(q.count)});
  }
  return {id:spec.id,players:v.players,full_tensor_cells:p.full_tensor_cells,required_cells:p.required_cells,profiles:allTotals,
    observed_payoff_spans:Object.fromEntries(v.players.map((id,i)=>[id,String(span[i])])),
    scope:'exact independent product-profile regret over every declared unilateral action at this fixed engine state and seed; local scale uses shared observed union payoff range; no omitted-action, correlated-profile, or multiplayer campaign guarantee'};
}
function truncate(weights,cap){
  assert(Number.isInteger(cap)&&cap>=1);
  const a=weights.map(int),total=sum(a);assert(total>0n);
  const sorted=a.map((w,i)=>({w,i})).filter(x=>x.w>0n).sort((x,y)=>x.w>y.w?-1:x.w<y.w?1:x.i-y.i);
  const keep=new Set(sorted.slice(0,cap).map(x=>x.i)),b=a.map((w,i)=>keep.has(i)?w:0n);
  return {weights:b.map(String),discarded_mass:frac(total-sum(b),total)};
}
export {plan,certify,truncate,frac};
