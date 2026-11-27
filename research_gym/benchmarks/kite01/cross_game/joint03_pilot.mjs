import assert from 'node:assert/strict';
import {plan,certify} from './sparse_joint.mjs';
import {rationalCmp} from './joint_search.mjs';
import {makeGame} from './joint_search_pilot.mjs';
import {search03} from './joint03.mjs';
const families=['coordination','anti_coordination','directed_cycle','general_sum'],modes=['br','hybrid','hybrid_prune'];
const average=a=>a.reduce((z,v)=>z+v,0)/a.length;
const value=f=>{const [n,d]=f.split('/').map(Number);return n/d};
async function pilot03(){
 const rows=[];
 for(let f=0;f<families.length;f++)for(let k=0;k<16;k++){
  const seed=7831201+100003*f+7919*k,{spec,oracle}=makeGame(families[f],seed);
  let s=(seed^0x7e419)>>>0;
  for(const p of spec.players){s^=s<<13;s^=s>>>17;s^=s<<5;const a=(s>>>0)%4;spec.profiles[0].weights[p]=Array.from({length:4},(_,i)=>i===a?'1':'0')}
  for(const mode of modes){
   const r=await search03(spec,oracle,{mode,budget:256,reserve:64,stall:4,maxSteps:40,batch:8});
   assert(r.unique_cells<=256&&r.unique_cells===r.new_cells);
   const final={...spec,profiles:[{id:'candidate',weights:r.profile}]},p=plan(final,256),cache=new Map(r.receipts.map(x=>[x.actions.join(','),x]));
   const q=certify(final,p.cells.map(a=>cache.get(a.join(','))),256).profiles[0];
   assert.deepEqual(q,r.certificate);assert(rationalCmp(r.final,r.start)<=0);
   for(const h of r.history.filter(x=>x.status==='pruned'))assert(rationalCmp(h.lower,h.incumbent)>0);
   rows.push({family:families[f],seed,spec,...r});
  }
 }
 const summary=[];
 for(const family of ['all',...families])for(const mode of modes){
  const a=rows.filter(r=>(family==='all'||r.family===family)&&r.mode===mode);
  summary.push({family,mode,n:a.length,start:average(a.map(r=>value(r.start))),final:average(a.map(r=>value(r.final))),exact:a.filter(r=>r.final==='0/1').length,le_005:a.filter(r=>rationalCmp(r.final,'1/20')<=0).length,le_020:a.filter(r=>rationalCmp(r.final,'1/5')<=0).length,mean_cells:average(a.map(r=>r.unique_cells)),mean_br_steps:average(a.map(r=>r.br_steps)),pruned:a.reduce((z,r)=>z+r.pruned,0),pruned_unqueried:a.reduce((z,r)=>z+r.pruned_unqueried,0),partial_checks:a.reduce((z,r)=>z+r.partial_checks,0),stop:Object.fromEntries([...new Set(a.map(r=>r.stop))].map(k=>[k,a.filter(r=>r.stop===k).length]))});
 }
 const paired=[];
 for(const mode of modes.slice(1))for(const baseline of (mode==='hybrid_prune'?['br','hybrid']:['br'])){
  const a=rows.filter(r=>r.mode===mode),b=new Map(rows.filter(r=>r.mode===baseline).map(r=>[r.spec.id,r]));
  const d=a.map(r=>({root:r.spec.id,gain:value(b.get(r.spec.id).final)-value(r.final),sign:rationalCmp(b.get(r.spec.id).final,r.final)}));
  paired.push({mode,baseline,wins:d.filter(x=>x.sign>0).length,losses:d.filter(x=>x.sign<0).length,ties:d.filter(x=>!x.sign).length,mean_gain:average(d.map(x=>x.gain))});
 }
 return {status:'development_only',roots:64,budget_per_arm:256,summary,paired,rows};
}
export {pilot03};
