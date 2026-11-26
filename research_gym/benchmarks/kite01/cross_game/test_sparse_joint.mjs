import assert from 'node:assert/strict';
import {plan,certify,truncate,frac} from './sparse_joint.mjs';
function make(sizes,weights){
 const players=sizes.map((_,i)=>'p'+i),rosters=Object.fromEntries(players.map((p,i)=>[p,Array.from({length:sizes[i]},(_,a)=>({id:'a'+a}))]));
 return {schema:'kite.sparse-joint.v1',id:'synthetic',players,rosters,denominator:100,profiles:[{id:'profile',weights:Object.fromEntries(players.map((p,i)=>[p,weights[i].map(String)]))}]};
}
function payoff(a,i,seed){let h=(seed+131*i+7)>>>0;for(let j=0;j<a.length;j++)h=(Math.imul(h^(a[j]+17*j+1),1664525)+1013904223)>>>0;return h%101}
const rec=(a,seed)=>({actions:a,payoffs:a.map((_,i)=>payoff(a,i,seed)),post_state_hash:'a'.repeat(64)});
function brute(sizes,weights,seed){
 const totals=weights.map(w=>w.reduce((a,b)=>a+BigInt(b),0n)),den=totals.reduce((a,b)=>a*b,100n),gain=[];
 for(let i=0;i<sizes.length;i++){
  let inc=0n;const dev=Array(sizes[i]).fill(0n);
  // Independent dense tensor enumeration: include zero-weight tuples, substitute one action in each tuple.
  function visit(a){if(a.length<sizes.length){for(let x=0;x<sizes[a.length];x++)visit([...a,x]);return}
   const w=a.reduce((z,x,j)=>z*BigInt(weights[j][x]),1n);inc+=w*BigInt(payoff(a,i,seed));
   for(let b=0;b<sizes[i];b++){const t=[...a];t[i]=b;dev[b]+=w*BigInt(payoff(t,i,seed))}
  }
  visit([]);gain.push(dev.reduce((a,b)=>a>b?a:b)-inc);
 }
 return gain.map(g=>frac(g,den));
}
function runTests(){
 let cases=0,playerChecks=0,rejected=0;
 const sizes=[2,3,2], options=[[1,0],[0,1],[1,1],[1,3],[3,1]],mid=[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,2],[3,2,1]];
 for(const x of options)for(const y of mid)for(const z of options)for(let seed=0;seed<4;seed++){
  const w=[x,y,z],spec=make(sizes,w),p=plan(spec),cells=p.cells.map(a=>rec(a,seed)),q=certify(spec,cells);
  assert.deepEqual(q.profiles[0].players.map(a=>a.regret_global),brute(sizes,w,seed));cases++;playerChecks+=3;
 }
 const s=Array(7).fill(4),pure=make(s,s.map(()=>[1,0,0,0])),two=make(s,s.map((_,i)=>i<2?[3,1,0,0]:[1,0,0,0])),allTwo=make(s,s.map(()=>[1,1,0,0])),dense=make(s,s.map(()=>[1,1,1,1]));
 assert.equal(plan(pure).required_cells,22);assert.equal(plan(two).required_cells,72);assert.equal(plan(allTwo).required_cells,1024);assert.throws(()=>plan(dense));
 assert.equal(plan(dense,16384).required_cells,16384);
 const union={...two,profiles:[two.profiles[0],{...pure.profiles[0],id:'pure'}]};const p=plan(union);assert.equal(p.required_cells,72);const cells=p.cells.map(a=>rec(a,8));const q=certify(union,cells);
 assert.equal(q.profiles.length,2);for(const broken of [cells.slice(1),[...cells.slice(1),cells[1]],[{...cells[0],payoffs:[101,...cells[0].payoffs.slice(1)]},...cells.slice(1)],[{...cells[0],post_state_hash:'fake'},...cells.slice(1)]]){assert.throws(()=>certify(union,broken));rejected++}
 assert.throws(()=>plan(two,71));assert.throws(()=>plan({...two,profiles:[{...two.profiles[0],weights:{...two.profiles[0].weights,p0:['0','0','0','0']}}]}));
 assert.deepEqual(truncate(['0','7','7','2'],2),{weights:['0','7','7','0'],discarded_mass:'1/8'});
 return {status:'PASS',scope:'synthetic exact finite-domain only',cases,playerChecks,rejected,pure_cells:22,two_mixed_cells:72,all_seven_mixed_cells:1024,full_tensor_cells:16384};
}
console.log(JSON.stringify(runTests()));
