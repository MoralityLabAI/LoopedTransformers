import assert from 'node:assert/strict';
import {makePlans,auditPlans} from './native_joint.mjs';
function run(){
 const players=['ARCHIVIST','BROKER','CANTOR','CONVENOR','HEGEMON','INFILTRATOR','STATE'];
 const rosters=Object.fromEntries(players.map(p=>[p,Array.from({length:4},(_,i)=>({id:p+':'+i,orders:i?[{id:p+':'+i}]:[]}))]));
 const h='a'.repeat(64),root={id:'mock:HEGEMON:STATE',actors:['HEGEMON','STATE'],denominator:42,source:{board_hash:h,campaign_id:'mock-seed'},actions:{row:rosters.HEGEMON,col:rosters.STATE},worlds:[{id:'all_pass',fixed_orders:Object.fromEntries(players.filter(p=>!['HEGEMON','STATE'].includes(p)).map(p=>[p,[]]))}]};
 const native={roots:[root],pure_profiles:[{players,board_hash:h,mode:'all_pass',rosters,denominator:42}]};
 const profile=[[['9','5','3','1'],['7','4','2','1']],[['3','7','1','5'],['1','4','0','2']],[['0','1','0','1'],['1','0','1','0']]];
 const report={profiles:[{id:root.id,actors:root.actors,scale:'local',belief:'common',rows:['full','workspace_full_readout','workspace_direct_readout'].map((arm,i)=>({arm,depth:32,profile:{x:profile[i][0],y:profile[i][1]}}))}]};
 const manifest=makePlans(native,report);assert.equal(manifest.jobs.length,1);
 const j=manifest.jobs[0];assert.equal(j.plan.required_cells,256);assert.equal(j.spec.profiles.length,7);
 const cells=j.plan.cells.map(t=>({actions:t,payoffs:players.map((_,i)=>t.reduce((z,a,k)=>(z*11+a*(k+1)+i+3)%43,i)),post_state_hash:h}));
 const receipts={schema:'kite.native-joint-receipts.v1',jobs:[{id:j.spec.id,digest:j.digest,board_hash:h,cells}]};
 const a=auditPlans(manifest,receipts);assert.equal(a.unique_native_cells,256);assert.equal(a.jobs[0].truncation_bounds.length,21);assert.equal(a.jobs[0].profiles.length,7);
 const bad=structuredClone(receipts);bad.jobs[0].digest='b'.repeat(64);assert.throws(()=>auditPlans(manifest,bad));
 const missing=structuredClone(receipts);missing.jobs[0].cells.pop();assert.throws(()=>auditPlans(manifest,missing));
 const wrong=structuredClone(manifest);wrong.jobs[0].spec.truncations[0].discarded_mass.HEGEMON='0/1';assert.throws(()=>auditPlans(wrong,receipts));
 return {status:'PASS',scope:'mock planner/receipt pipeline',profiles:7,players:7,unique_cells:256,truncation_bound_checks:21,rejected:3};
}
console.log(JSON.stringify(run()));
