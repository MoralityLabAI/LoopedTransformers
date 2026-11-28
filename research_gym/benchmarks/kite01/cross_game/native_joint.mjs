import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {readFileSync,writeFileSync} from 'node:fs';
import {plan,certify,truncate,frac} from './sparse_joint.mjs';
const digest=v=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
const equalOrders=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const arms=['full','workspace_full_readout','workspace_direct_readout'];
const budget=256;
function makePlans(native,report){
 const jobs=[];
 for(const root of native.roots){
  const focal=root.actors;
  if(focal.join(':')!=='FRANCE:GERMANY'&&focal.join(':')!=='HEGEMON:STATE')continue;
  const pure=native.pure_profiles.find(p=>p.board_hash===root.source.board_hash&&(p.mode==='all_hold'||p.mode==='all_pass'));
  assert(pure,'missing seven-player anchor');
  const world=root.worlds[0];
  assert(world.id===pure.mode&&world.fixed_orders,'unexpected world');
  const players=pure.players,rosters=Object.fromEntries(players.map(p=>[p,p===focal[0]?root.actions.row:p===focal[1]?root.actions.col:pure.rosters[p]]));
  assert.equal(players.length,7);assert.equal(root.denominator,pure.denominator);
  const anchor={};for(const p of players){
   if(focal.includes(p))anchor[p]=0;
   else {const a=rosters[p].findIndex(a=>equalOrders(a.orders,world.fixed_orders[p]));assert(a>=0,'fixed world order absent from roster');anchor[p]=a}
  }
  const rp=report.profiles.find(r=>r.id===root.id&&r.scale==='local'&&r.belief==='common');
  assert(rp&&rp.actors.join(':')===focal.join(':'));
  const one=p=>rosters[p].map((_,i)=>i===anchor[p]?'1':'0');
  const profiles=[{id:'all_pure',weights:Object.fromEntries(players.map(p=>[p,one(p)]))}],truncations=[];
  for(const arm of arms){
   const q=rp.rows.find(r=>r.arm===arm&&r.depth===32);assert(q,'missing frozen pairwise profile');
   const weights=Object.fromEntries(players.map(p=>[p,p===focal[0]?q.profile.x:p===focal[1]?q.profile.y:one(p)]));
   profiles.push({id:arm,weights});
   const trimmed=Object.fromEntries(players.map(p=>[p,truncate(weights[p],2)]));
   profiles.push({id:arm+'_top2',weights:Object.fromEntries(players.map(p=>[p,trimmed[p].weights]))});
   truncations.push({source:arm,target:arm+'_top2',discarded_mass:Object.fromEntries(players.map(p=>[p,trimmed[p].discarded_mass]))});
  }
  const spec={schema:'kite.sparse-joint.v1',id:root.id+':joint',players,rosters,denominator:root.denominator,profiles,
    source:{...root.source,conditional_root_id:root.id,focal,background_world:world.id,profile_source:'local common belief, raw depth-32 profiles'},truncations};
  const p=plan(spec,budget);
  jobs.push({spec,plan:p,digest:digest({spec,plan:p})});
 }
 assert(jobs.length>0&&new Set(jobs.map(j=>j.spec.id)).size===jobs.length);
 return {schema:'kite.native-joint-plan.v1',budget,jobs};
}
const parts=s=>s.split('/').map(BigInt);
function productRetained(masses){let n=1n,d=1n;for(const m of masses){const [a,b]=parts(m);assert(b>0n&&a>=0n&&a<b);n*=b-a;d*=b}return [d-n,d]}
function plusBound(a,b){let n=a[0]*b[1]+b[0]*a[1],d=a[1]*b[1];if(n>d)n=d;return [n,d]}
function auditPlans(manifest,native){
 assert.equal(manifest.schema,'kite.native-joint-plan.v1');assert.equal(native.schema,'kite.native-joint-receipts.v1');assert.equal(manifest.budget,budget);
 assert.equal(native.jobs.length,manifest.jobs.length);assert.equal(new Set(native.jobs.map(j=>j.id)).size,native.jobs.length);
 const results=[];
 for(const j of manifest.jobs){
  assert.deepEqual(plan(j.spec,budget),j.plan);assert.equal(digest({spec:j.spec,plan:j.plan}),j.digest);
  const r=native.jobs.find(x=>x.id===j.spec.id);assert(r&&r.digest===j.digest&&r.board_hash===j.spec.source.board_hash,'native receipt scope mismatch');
  const cert=certify(j.spec,r.cells,budget),bounds=[];
  for(const t of j.spec.truncations){
   const a=cert.profiles.find(q=>q.id===t.source),b=cert.profiles.find(q=>q.id===t.target);
   const pa=j.spec.profiles.find(q=>q.id===t.source),pb=j.spec.profiles.find(q=>q.id===t.target);
   for(const p of j.spec.players){const q=truncate(pa.weights[p],2);assert.deepEqual(pb.weights[p],q.weights);assert.equal(t.discarded_mass[p],q.discarded_mass)}
   const full=productRetained(j.spec.players.map(p=>t.discarded_mass[p]));
   for(const p of j.spec.players){
    const opp=productRetained(j.spec.players.filter(x=>x!==p).map(x=>t.discarded_mass[x])),bound=plusBound(full,opp);
    const [x,xd]=parts(a.players.find(v=>v.player===p).regret_global),[y,yd]=parts(b.players.find(v=>v.player===p).regret_global);
    const raw=x*yd-y*xd,abs=raw<0n?-raw:raw;
    assert(abs*bound[1]<=bound[0]*xd*yd,'truncation TV bound violated');
    bounds.push({source:t.source,target:t.target,player:p,discarded_mass:t.discarded_mass[p],total_tv:frac(...full),opponent_tv:frac(...opp),regret_change_bound:frac(...bound),observed_absolute_change:frac(abs,xd*yd)});
   }
  }
  results.push({...cert,source:j.spec.source,plan_profiles:j.plan.profiles,truncation_bounds:bounds});
 }
 return {schema:'kite.native-joint-audit.v1',scope:'development-only fixed background, fixed seed, seven declared action rosters; no independent campaign generalization',jobs:results,unique_native_cells:results.reduce((s,r)=>s+r.required_cells,0),campaign_clusters:new Set(results.map(r=>r.source.campaign_id)).size};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
if(process.argv[2]==='plan'){
 const j=makePlans(JSON.parse(readFileSync(process.argv[3],'utf8')),JSON.parse(readFileSync(process.argv[4],'utf8')));
 writeFileSync(process.argv[5],JSON.stringify(j,null,2)+'\n');console.log('JOINT_PLAN '+JSON.stringify(j.jobs.map(x=>({id:x.spec.id,cells:x.plan.required_cells,full:x.plan.full_tensor_cells,profiles:x.plan.profiles}))));
}else if(process.argv[2]==='audit'){
 const j=auditPlans(JSON.parse(readFileSync(process.argv[3],'utf8')),JSON.parse(readFileSync(process.argv[4],'utf8')));
 writeFileSync(process.argv[5],JSON.stringify(j,null,2)+'\n');console.log('JOINT_NATIVE_SUMMARY '+JSON.stringify({cells:j.unique_native_cells,clusters:j.campaign_clusters,jobs:j.jobs.map(r=>({id:r.id,cells:r.required_cells,profiles:r.profiles.map(p=>({id:p.id,global:p.global_max_regret,local:p.local_max_regret_observed_range,exact:p.restricted_exact_nash,profitable:p.players.filter(x=>x.profitable).map(x=>x.player)}))}))}));
}else throw Error('expected plan or audit');
}
export {makePlans,auditPlans};
