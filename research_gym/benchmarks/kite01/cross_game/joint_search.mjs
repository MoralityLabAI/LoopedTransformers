import assert from 'node:assert/strict';
import {plan,certify,frac} from './sparse_joint.mjs';

const parts=s=>s.split('/').map(BigInt);
const sum=a=>a.reduce((z,v)=>z+v,0n);
const prod=a=>a.reduce((z,v)=>z*v,1n);
const key=a=>a.join(',');
const nz=a=>a.flatMap((w,i)=>BigInt(w)>0n?[i]:[]);
const rationalCmp=(x,y)=>{const [a,b]=parts(x),[c,d]=parts(y),z=a*d-c*b;return z<0n?-1:z>0n?1:0};
function add(x,y){const [a,b]=parts(x),[c,d]=parts(y);return frac(a*d+c*b,b*d)}
function profileKey(w,players){return players.map(p=>w[p].map(x=>frac(BigInt(x),sum(w[p].map(BigInt)))).join(',')).join(';')}
function lex(c,d){
 const r=rationalCmp(c.global_max_regret,d.global_max_regret);
 return r||rationalCmp(c.players.reduce((z,p)=>add(z,p.regret_global),'0/1'),d.players.reduce((z,p)=>add(z,p.regret_global),'0/1'));
}
function partialBounds(spec,receipts,limit=4096){
 const p=plan(spec,limit),required=new Set(p.cells.map(key)),known=new Map(),D=BigInt(spec.denominator);
 assert(D>0n);
 for(const r of receipts){
  assert(Array.isArray(r.actions)&&r.actions.length===spec.players.length&&r.actions.every(Number.isInteger),'tuple');
  const k=key(r.actions);assert(required.has(k)&&!known.has(k),'unexpected or duplicate receipt');
  assert(Array.isArray(r.payoffs)&&r.payoffs.length===spec.players.length&&r.payoffs.every(v=>(typeof v==='string'&&/^\d+$/.test(v)||Number.isSafeInteger(v))&&BigInt(v)>=0n&&BigInt(v)<=D),'payoff bounds');
  assert(typeof r.post_state_hash==='string'&&/^[a-f0-9]{64}$/.test(r.post_state_hash),'hash');
  known.set(k,r.payoffs.map(BigInt));
 }
 const result=[];
 for(const profile of spec.profiles){
  const w=spec.players.map(p=>profile.weights[p].map(BigInt)),totals=w.map(sum),den=prod(totals)*D;
  const players=[];
  let low=0n,high=0n;
  for(let i=0;i<w.length;i++){
   let lo=0n,hi=0n;const deviations=[];
   for(let b=0;b<w[i].length;b++){
    let l=0n,h=0n;
    for(const t of p.cells){
     let c=totals[i]*(t[i]===b?1n:0n)-w[i][t[i]];
     if(c===0n)continue;
     for(let j=0;j<w.length;j++)if(j!==i)c*=w[j][t[j]];
     if(c===0n)continue;
     const v=known.get(key(t));
     if(v){l+=c*v[i];h+=c*v[i]}else if(c>0n)h+=c*D;else l+=c*D;
    }
    if(l>lo)lo=l;if(h>hi)hi=h;
    deviations.push({action:b,lower:frac(l>0n?l:0n,den),upper:frac(h>0n?h:0n,den)});
   }
   if(lo>low)low=lo;if(hi>high)high=hi;
   players.push({player:spec.players[i],lower:frac(lo,den),upper:frac(hi,den),deviations});
  }
  result.push({id:profile.id,lower:frac(low,den),upper:frac(high,den),certified_le_020:5n*high<=den,refuted_le_020:5n*low>den,players});
 }
 return {required_cells:p.required_cells,known_cells:known.size,profiles:result};
}
function candidateMoves(spec,current,cert,mode,maxMixed){
 const w=current.weights,players=spec.players;
 const order=cert.players.map((v,i)=>({v,i})).sort((a,b)=>rationalCmp(b.v.regret_global,a.v.regret_global)||a.i-b.i),out=[];
 for(const {v,i} of order){
  if(!v.profitable)continue;
  const p=players[i],n=w[p].length,b=spec.rosters[p].findIndex(a=>a.id===v.best_action);assert(b>=0);
  const original=w[p].map(BigInt);
  const bestOwn=original.reduce((j,x,k)=>x>original[j]?k:j,0);
  const pure=Array(n).fill('0');pure[b]='1';
  out.push({player:p,kind:'pure',weights:{...w,[p]:pure}});
  if(mode==='mixed'&&bestOwn!==b){
   const mixedOther=players.filter(q=>q!==p&&nz(w[q]).length>1).length;
   if(mixedOther<maxMixed){
    for(const [a,z] of [[1,1],[3,1],[1,3]]){
     const mix=Array(n).fill('0');mix[bestOwn]=String(a);mix[b]=String(z);
     out.push({player:p,kind:'blend-'+a+':'+z,weights:{...w,[p]:mix}});
    }
   }
  }
 }
 return out;
}
// Deterministic exact-search policy. Every candidate admitted in full before comparison;
// no selection from uncertified partial profiles. Budget counts unique oracle cells.
async function search(spec,oracle,{mode='mixed',budget=256,maxMixed=2,maxRounds=12,initialReceipts=[]}={}){
 assert(['br','pure','mixed'].includes(mode)&&Number.isInteger(budget)&&budget>=1&&budget<=4096);
 assert(Number.isInteger(maxMixed)&&maxMixed>=0&&maxMixed<=2&&Number.isInteger(maxRounds)&&maxRounds>0);
 assert(spec.profiles.length===1,'one initial profile');
 const initial=spec.profiles[0],players=spec.players;
 assert(players.every(p=>nz(initial.weights[p]).length===1),'pure start for matched baseline');
 const cache=new Map(),history=[],seen=new Map();let newCalls=0;
 function readReceipt(r){
  assert(r&&Array.isArray(r.actions)&&r.actions.length===players.length&&r.actions.every((v,i)=>Number.isInteger(v)&&v>=0&&v<spec.rosters[players[i]].length),'bad receipt tuple');
  assert(Array.isArray(r.payoffs)&&r.payoffs.length===players.length&&r.payoffs.every(v=>Number.isSafeInteger(v)&&v>=0&&BigInt(v)<=BigInt(spec.denominator)||typeof v==='string'&&/^\d+$/.test(v)&&BigInt(v)<=BigInt(spec.denominator)),'bad receipt payoffs');
  assert(/^[a-f0-9]{64}$/.test(r.post_state_hash),'bad receipt digest');
  const k=key(r.actions);assert(!cache.has(k),'duplicate receipt');cache.set(k,r);
 }
 for(const r of initialReceipts)readReceipt(r);assert(cache.size<=budget);
 async function evaluate(weights,tag,round){
  const k=profileKey(weights,players);
  if(seen.has(k))return {...seen.get(k),cachedProfile:true};
  const profile={id:'candidate',weights},s={...spec,profiles:[profile]};
  let frontier;
  // Refuse over-budget candidates before any query. Distinguish budget error from malformed spec.
  try{frontier=plan(s,budget)}catch(e){if(/exceeds payoff budget/.test(String(e)))return null;throw e}
  const missing=frontier.cells.filter(a=>!cache.has(key(a)));
  if(cache.size+missing.length>budget)return null;
  if(missing.length){
   const rows=await oracle(missing);
   assert(Array.isArray(rows)&&rows.length===missing.length,'oracle batch count');
   const wanted=new Set(missing.map(key));
   for(const r of rows){assert(wanted.delete(key(r.actions)),'unexpected oracle cell');readReceipt(r)}
   assert(wanted.size===0);
   newCalls+=missing.length;
  }
  const cert=certify(s,frontier.cells.map(t=>cache.get(key(t))),budget).profiles[0];
  const result={weights,cert};seen.set(k,result);
  history.push({tag,round,new_cells:missing.length,cache_cells:cache.size,required_cells:frontier.required_cells,weights,regret:cert.global_max_regret});
  return result;
 }
 let current=await evaluate(initial.weights,'initial',0);assert(current,'initial frontier exceeds budget');
 const start=current.cert.global_max_regret;let best=current,stop='round_limit',accepted=0;
 const visited=new Set([profileKey(current.weights,players)]);
 for(let round=1;round<=maxRounds;round++){
  if(current.cert.restricted_exact_nash){stop='zero_regret';break}
  const proposals=candidateMoves(spec,current,current.cert,mode,maxMixed);
  // BR follows largest profitable deviation even if maximum regret rises; retains best certified profile.
  if(mode==='br'){
   const a=proposals[0];if(!a){stop='no_proposal';break}
   const next=await evaluate(a.weights,a.kind+':'+a.player,round);
   if(!next){stop='budget';break}
   const k=profileKey(next.weights,players);if(visited.has(k)){stop='cycle';break}
   visited.add(k);current=next;accepted++;
   if(lex(current.cert,best.cert)<0)best=current;
   continue;
  }
  let next=current,admitted=0,refused=0;
  for(const a of proposals){
   const q=await evaluate(a.weights,a.kind+':'+a.player,round);
   if(!q){refused++;continue}
   admitted++;
   if(lex(q.cert,best.cert)<0)best=q;
   if(lex(q.cert,next.cert)<0)next=q;
   if(best.cert.restricted_exact_nash)break;
  }
  if(next===current||lex(next.cert,current.cert)>=0){stop=admitted===0&&refused?'budget':'local_minimum';break}
  current=next;accepted++;
 }
 return {mode,budget,start,final:best.cert.global_max_regret,profile:best.weights,certificate:best.cert,
   unique_cells:cache.size,new_cells:newCalls,initial_cells:initialReceipts.length,evaluated_profiles:seen.size,accepted,stop,
   history,receipts:[...cache.values()]};
}
export {search,partialBounds,rationalCmp};
