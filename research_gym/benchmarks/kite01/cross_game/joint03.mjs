import assert from 'node:assert/strict';
import {plan,certify,frac} from './sparse_joint.mjs';
import {partialBounds,rationalCmp} from './joint_search.mjs';

const sum=a=>a.reduce((s,v)=>s+v,0n);
const product=a=>a.reduce((s,v)=>s*v,1n);
const key=a=>a.join(',');
const nz=a=>a.filter(v=>BigInt(v)>0n).length;
function profileKey(w,players){return players.map(p=>w[p].map(x=>frac(BigInt(x),sum(w[p].map(BigInt)))).join(',')).join(';')}
function add(x,y){const [a,b]=x.split('/').map(BigInt),[c,d]=y.split('/').map(BigInt);return frac(a*d+c*b,b*d)}
function cmp(a,b){return rationalCmp(a.global_max_regret,b.global_max_regret)||rationalCmp(a.players.reduce((z,p)=>add(z,p.regret_global),'0/1'),b.players.reduce((z,p)=>add(z,p.regret_global),'0/1'))}
function pureWeights(w,i){return w.map((_,j)=>j===i?'1':'0')}
function proposalList(spec,current){
 const players=spec.players,w=current.weights,out=[];
 const order=current.cert.players.map((p,i)=>({p,i})).sort((a,b)=>rationalCmp(b.p.regret_global,a.p.regret_global)||a.i-b.i);
 for(const {p,i} of order){
  if(!p.profitable)continue;
  const id=players[i],a=spec.rosters[id].findIndex(x=>x.id===p.best_action);
  assert(a>=0);
  const modal=w[id].map(BigInt).reduce((k,v,j,arr)=>v>arr[k]?j:k,0);
  const pure=pureWeights(w[id],a);
  out.push({tag:'pure:'+id,weights:{...w,[id]:pure}});
  if(a!==modal&&players.filter(x=>x!==id&&nz(w[x])>1).length<2){
   for(const [m,n] of [[1,1],[3,1],[1,3]]){
    const z=w[id].map(()=> '0');z[modal]=String(m);z[a]=String(n);
    out.push({tag:'blend:'+id+':'+m+':'+n,weights:{...w,[id]:z}});
   }
  }
 }
 return out;
}
// Max absolute signed gain coefficient across players/deviations, scaled by common profile mass.
// Query high-leverage cells first in the return-only completion phase.
function priority(weights,players,tuple){
 const w=players.map(p=>weights[p].map(BigInt)),S=w.map(sum);
 let v=0n;
 for(let i=0;i<w.length;i++){
  let others=1n;for(let j=0;j<w.length;j++)if(j!==i)others*=w[j][tuple[j]];
  if(!others)continue;
  const own=w[i][tuple[i]],c=(S[i]-own>own?S[i]-own:own)*others;
  if(c>v)v=c;
 }
 return v;
}
async function search03(spec,oracle,{mode='br',budget=256,reserve=64,stall=4,maxSteps=40,batch=8}={}){
 assert(['br','hybrid','hybrid_prune'].includes(mode)&&Number.isSafeInteger(budget)&&budget>=1&&budget<=4096);
 assert(Number.isSafeInteger(reserve)&&reserve>=0&&reserve<budget&&Number.isSafeInteger(stall)&&stall>=1&&Number.isSafeInteger(maxSteps)&&maxSteps>=1&&Number.isSafeInteger(batch)&&batch>=1);
 assert(spec.profiles.length===1);
 const players=spec.players,start=spec.profiles[0];assert(players.every(p=>nz(start.weights[p])===1));
 const cache=new Map(),complete=new Map(),history=[];
 let calls=0,pruned=0,partialChecks=0,prunedUnqueried=0,attempts=0,transitions=0;
 async function query(tuples){
  if(!tuples.length)return;
  assert(cache.size+tuples.length<=budget);
  const want=new Set(tuples.map(key)),rows=await oracle(tuples);assert(rows.length===tuples.length);
  for(const r of rows){
   const k=key(r.actions);assert(want.delete(k)&&!cache.has(k),'oracle tuple mismatch');
   assert(r.actions.length===players.length&&r.actions.every((x,i)=>Number.isInteger(x)&&x>=0&&x<spec.rosters[players[i]].length));
   assert(Array.isArray(r.payoffs)&&r.payoffs.length===players.length&&r.payoffs.every(v=>(typeof v==='string'&&/^\d+$/.test(v)||Number.isSafeInteger(v))&&BigInt(v)>=0n&&BigInt(v)<=BigInt(spec.denominator)));
   assert(/^[a-f0-9]{64}$/.test(r.post_state_hash));
   cache.set(k,r);calls++;
  }
  assert(want.size===0);
 }
 async function evaluate(weights,tag,cap,incumbent=null,prune=false){
  attempts++;
  const k=profileKey(weights,players);
  if(complete.has(k))return {state:complete.get(k),status:'cached'};
  const s={...spec,profiles:[{id:'candidate',weights}]};
  let frontier;
  try{frontier=plan(s,budget)}catch(e){if(/exceeds payoff budget/.test(String(e)))return {status:'budget'};throw e}
  let missing=frontier.cells.filter(t=>!cache.has(key(t)));
  // Full admission required unless partial bounds can rule out a return-only candidate.
  // A partially queried candidate never becomes an exploratory state or certified output.
  if(!prune&&cache.size+missing.length>cap)return {status:'budget'};
  const before=cache.size;
  if(prune)missing.sort((a,b)=>{const x=priority(weights,players,a),y=priority(weights,players,b);return x>y?-1:x<y?1:0});
  while(missing.length){
   if(prune&&incumbent){
    const known=frontier.cells.filter(t=>cache.has(key(t))).map(t=>cache.get(key(t)));
    const bounds=partialBounds(s,known,budget).profiles[0];partialChecks++;
    if(rationalCmp(bounds.lower,incumbent.cert.global_max_regret)>0){
     pruned++;prunedUnqueried+=missing.length;
     history.push({tag,status:'pruned',weights,lower:bounds.lower,upper:bounds.upper,incumbent:incumbent.cert.global_max_regret,required_cells:frontier.required_cells,new_cells:cache.size-before,unqueried:missing.length,cache_cells:cache.size});
     return {status:'pruned'};
    }
   }
   const count=Math.min(missing.length,cap-cache.size,prune?batch:missing.length);
   if(count<=0){history.push({tag,status:'budget_partial',weights,new_cells:cache.size-before,unqueried:missing.length,cache_cells:cache.size});return {status:'budget'}}
   await query(missing.splice(0,count));
  }
  const cert=certify(s,frontier.cells.map(t=>cache.get(key(t))),budget).profiles[0];
  const state={weights,cert};complete.set(k,state);
  history.push({tag,status:'complete',weights,regret:cert.global_max_regret,required_cells:frontier.required_cells,new_cells:cache.size-before,cache_cells:cache.size});
  return {status:'complete',state};
 }
 let r=await evaluate(start.weights,'initial',budget);assert(r.state,'initial frontier exceeds budget');
 let workspace=r.state,best=r.state,brSteps=0,stale=0,stop='step_limit',trigger='none';
 const startRegret=best.cert.global_max_regret,visited=new Set([profileKey(workspace.weights,players)]);
 async function brStep(cap){
  if(workspace.cert.restricted_exact_nash)return 'zero_regret';
  const a=proposalList(spec,workspace)[0];if(!a)return 'no_proposal';
  const k=profileKey(a.weights,players);if(visited.has(k))return 'cycle';
  const next=await evaluate(a.weights,'explore:'+a.tag,cap);
  if(!next.state)return next.status;
  workspace=next.state;visited.add(k);transitions++;brSteps++;
  if(cmp(workspace.cert,best.cert)<0){best=workspace;stale=0}else stale++;
  return 'continue';
 }
 // Same sequential BR workspace in all arms; hybrid diverts only after stalled/cycled
 // or reaching the reserved-budget boundary. The certified best state is independent.
 while(brSteps<maxSteps){
  if(mode!=='br'&&stale>=stall){trigger='stagnation';break}
  const q=await brStep(mode==='br'?budget:Math.max(cache.size,budget-reserve));
  if(q!=='continue'){trigger=q;break}
 }
 if(best.cert.restricted_exact_nash){stop='zero_regret'}
 else if(mode==='br'){stop=trigger==='none'?'step_limit':trigger}
 else {
  // One return-only sweep from the current best. No pruning of BR exploration.
  const base=best,proposals=proposalList(spec,base);
  for(const a of proposals){
   if(best.cert.restricted_exact_nash)break;
   const q=await evaluate(a.weights,'return:'+a.tag,budget,best,mode==='hybrid_prune');
   if(q.state&&cmp(q.state.cert,best.cert)<0)best=q.state;
  }
  // Resume original exploratory BR path if budget remains. This is essential:
  // a return-only branch does not replace the workspace or force monotonicity.
  while(brSteps<maxSteps&&!best.cert.restricted_exact_nash){
   const q=await brStep(budget);if(q!=='continue'){stop=q;break}
  }
  if(best.cert.restricted_exact_nash)stop='zero_regret';
 }
 return {mode,budget,reserve,stall,maxSteps,batch,start:startRegret,final:best.cert.global_max_regret,profile:best.weights,certificate:best.cert,workspace:workspace.weights,workspace_regret:workspace.cert.global_max_regret,unique_cells:cache.size,new_cells:calls,attempts,evaluated_profiles:complete.size,br_steps:brSteps,transitions,trigger,stop,pruned,partial_checks:partialChecks,pruned_unqueried:prunedUnqueried,history,receipts:[...cache.values()]};
}
export {search03,priority};
