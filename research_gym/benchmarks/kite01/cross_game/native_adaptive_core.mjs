import assert from 'node:assert/strict';
import {search03} from './joint03.mjs';
import {plan,certify,frac} from './sparse_joint.mjs';
import {rationalCmp} from './joint_search.mjs';

const modes=['br','hybrid','hybrid_prune'],budget=256;
const key=a=>a.join(',');
const ratio=s=>{const [n,d]=s.split('/').map(Number);return n/d};
async function runBoard(spec,oracle){
 const results=[],union=new Map(),D=BigInt(spec.denominator);
 assert(spec.players.length===7);
 for(const mode of modes){
  const r=await search03(spec,t=>oracle(mode,t),{mode,budget,reserve:64,stall:4,maxSteps:40,batch:8});
  assert(r.unique_cells===r.new_cells&&r.unique_cells<=budget);
  const local=new Map();
  for(const v of r.receipts){
   const k=key(v.actions);assert(!local.has(k));local.set(k,v);
   if(union.has(k))assert.deepEqual(v,union.get(k),'same frozen cell changed across arms');else union.set(k,v);
  }
  const final={...spec,profiles:[{id:'candidate',weights:r.profile}]},p=plan(final,budget);
  assert.deepEqual(certify(final,p.cells.map(t=>local.get(key(t))),budget).profiles[0],r.certificate);
  for(const h of r.history.filter(x=>x.status==='pruned'))assert(rationalCmp(h.lower,h.incumbent)>0);
  results.push(r);
 }
 const spans=spec.players.map((_,i)=>{
  let lo=D,hi=0n;for(const v of union.values()){const z=BigInt(v.payoffs[i]);if(z<lo)lo=z;if(z>hi)hi=z}return hi-lo;
 });
 for(const r of results){
  const player_local=r.certificate.players.map((p,i)=>{const [a,b]=p.regret_utility_units.split('/').map(BigInt);assert(spans[i]>0n||a===0n);return {player:p.player,regret:frac(a,b*(spans[i]||1n))}});
  const local=player_local.reduce((z,p)=>rationalCmp(p.regret,z)>0?p.regret:z,'0/1');
  r.shared_observed_range={max_regret:local,players:player_local,pass_020:rationalCmp(local,'1/5')<=0};
 }
 return {spec,results,union_cells:union.size,shared_spans:Object.fromEntries(spec.players.map((p,i)=>[p,String(spans[i])]))};
}
function summarize(boards){
 const n=boards.length,mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
 assert(n>0);
 const summary=modes.map(mode=>{
  const a=boards.map(b=>b.results.find(x=>x.mode===mode));return {mode,n,mean_start:mean(a.map(r=>ratio(r.start))),mean_final:mean(a.map(r=>ratio(r.final))),mean_shared_local:mean(a.map(r=>ratio(r.shared_observed_range.max_regret))),exact:a.filter(r=>r.certificate.restricted_exact_nash).length,local_le_020:a.filter(r=>r.shared_observed_range.pass_020).length,mean_calls:mean(a.map(r=>r.unique_cells)),pruned:a.reduce((s,r)=>s+r.pruned,0)}
 });
 const paired=modes.slice(1).map(mode=>{
  const d=boards.map(b=>{const a=b.results.find(r=>r.mode===mode),c=b.results.find(r=>r.mode==='br');return {id:b.spec.id,sign:rationalCmp(c.final,a.final),gain:ratio(c.final)-ratio(a.final)}});
  return {mode,base:'br',wins:d.filter(x=>x.sign>0).length,losses:d.filter(x=>x.sign<0).length,ties:d.filter(x=>!x.sign).length,mean_gain:mean(d.map(x=>x.gain))};
 });
 return {status:'native_development_only',boards:n,campaign_clusters:new Set(boards.map(b=>b.spec.source.campaign_id)).size,summary,paired,scope:'fixed integer tactical utility, declared legal action rosters, frozen engine and seed; observed local range shared across arms, not full action tensor'};
}
export {runBoard,summarize};
