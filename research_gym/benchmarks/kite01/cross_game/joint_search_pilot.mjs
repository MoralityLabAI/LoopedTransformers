import assert from 'node:assert/strict';
import {plan,certify} from './sparse_joint.mjs';
import {search,partialBounds,rationalCmp} from './joint_search.mjs';
function rng(seed){let s=seed>>>0;return n=>{s^=s<<13;s^=s>>>17;s^=s<<5;return Math.floor((s>>>0)*n/4294967296)}}
function makeGame(family,seed,n=7,m=4){
 const r=rng(seed),players=Array.from({length:n},(_,i)=>'p'+i),rosters=Object.fromEntries(players.map(p=>[p,Array.from({length:m},(_,a)=>({id:'a'+a}))]));
 const bias=players.map(()=>Array.from({length:m},()=>r(21)));
 const tables=players.map(()=>Array.from({length:m},()=>Array.from({length:m},()=>r(41))));
 const target=players.map(()=>r(m));
 function utility(t,i){
  const a=t[i],l=t[(i+n-1)%n],q=t[(i+1)%n],b=bias[i][a];
  if(family==='coordination')return b+35*Number(a===l)+35*Number(a===q)+10*Number(a===target[i]);
  if(family==='anti_coordination')return b+35*Number(a!==l)+35*Number(a!==q)+10*Number(a===target[i]);
  if(family==='directed_cycle')return b+65*Number(a===(q+1)%m)+15*Number(a===l);
  if(family==='general_sum')return b+tables[i][a][l]+tables[i][q][a];
  throw Error('family');
 }
 const spec={schema:'kite.sparse-joint.v1',id:family+':'+seed,players,rosters,denominator:100,profiles:[{id:'start',weights:Object.fromEntries(players.map(p=>[p,['1',...Array(m-1).fill('0')]]))}]};
 const oracle=async tuples=>tuples.map(t=>({actions:t,payoffs:players.map((_,i)=>utility(t,i)),post_state_hash:'a'.repeat(64)}));
 return {spec,oracle};
}
async function boundsTest(){
 let cases=0,checks=0;
 for(let seed=1;seed<=24;seed++){
  const {spec,oracle}=makeGame('general_sum',seed,3,3);
  spec.profiles[0].weights={p0:['2','1','0'],p1:['0','1','0'],p2:['1','3','2']};
  const p=plan(spec),cells=await oracle(p.cells),exact=certify(spec,cells).profiles[0];
  let prevLow='0/1',prevHigh='1/1';
  for(let k=0;k<=cells.length;k++){
   const q=partialBounds(spec,cells.slice(0,k)).profiles[0];
   assert(rationalCmp(q.lower,exact.global_max_regret)<=0&&rationalCmp(exact.global_max_regret,q.upper)<=0);
   assert(rationalCmp(prevLow,q.lower)<=0&&rationalCmp(prevHigh,q.upper)>=0);
   for(let i=0;i<q.players.length;i++){assert(rationalCmp(q.players[i].lower,exact.players[i].regret_global)<=0&&rationalCmp(exact.players[i].regret_global,q.players[i].upper)<=0);checks++}
   prevLow=q.lower;prevHigh=q.upper;cases++;
  }
  assert.equal(prevLow,exact.global_max_regret);assert.equal(prevHigh,exact.global_max_regret);
 }
 return {cases,player_bound_checks:checks};
}
async function pilot(){
 const rows=[];
 for(const family of ['coordination','anti_coordination','directed_cycle','general_sum'])for(let k=0;k<12;k++){
  const seed=601113+7919*k+100003*['coordination','anti_coordination','directed_cycle','general_sum'].indexOf(family),{spec,oracle}=makeGame(family,seed);
  for(const mode of ['br','pure','mixed']){
   const r=await search(spec,oracle,{mode,budget:256,maxMixed:2,maxRounds:12});
   assert(r.unique_cells<=256&&r.new_cells===r.unique_cells);
   const p={...spec,profiles:[{id:'candidate',weights:r.profile}]};
   const frontier=plan(p,256),stored=new Map(r.receipts.map(v=>[v.actions.join(','),v]));
   const q=certify(p,frontier.cells.map(a=>stored.get(a.join(','))),256).profiles[0];
   assert.equal(q.global_max_regret,r.final);assert(rationalCmp(r.final,r.start)<=0);
   rows.push({family,seed,spec,...r});
  }
 }
 const value=f=>{const [a,b]=f.split('/').map(Number);return a/b},mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
 const summary=[];
 for(const family of ['all','coordination','anti_coordination','directed_cycle','general_sum'])for(const mode of ['br','pure','mixed']){
  const a=rows.filter(r=>(family==='all'||r.family===family)&&r.mode===mode);
  summary.push({family,mode,n:a.length,mean_start:mean(a.map(r=>value(r.start))),mean_final:mean(a.map(r=>value(r.final))),mean_cells:mean(a.map(r=>r.unique_cells)),exact:a.filter(r=>r.final==='0/1').length,le_005:a.filter(r=>rationalCmp(r.final,'1/20')<=0).length,le_020:a.filter(r=>rationalCmp(r.final,'1/5')<=0).length,mean_candidates:mean(a.map(r=>r.evaluated_profiles)),stop:Object.fromEntries([...new Set(a.map(r=>r.stop))].map(k=>[k,a.filter(r=>r.stop===k).length]))});
 }
 const paired=['br','pure'].map(mode=>{const a=rows.filter(r=>r.mode==='mixed'),b=new Map(rows.filter(r=>r.mode===mode).map(r=>[r.spec.id,r]));const d=a.map(r=>({id:r.spec.id,delta:value(b.get(r.spec.id).final)-value(r.final),sign:rationalCmp(b.get(r.spec.id).final,r.final)}));return {baseline:mode,wins:d.filter(x=>x.sign>0).length,losses:d.filter(x=>x.sign<0).length,ties:d.filter(x=>!x.sign).length,mean_regret_gain:mean(d.map(x=>x.delta))}});
 return {status:'development_only',roots:48,budget_per_arm:256,summary,paired,rows};
}
export {makeGame,boundsTest,pilot};
