// Engine-only sparse mixed-strategy frontier; no live harness or pact enforcement.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {fresh,signature,hash,utility}=require('./export_theysing.cjs');
function main(){
 const {TheySingEngine}=require(path.resolve(process.argv[2])),m=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
 assert(m.schema==='kite.native-joint-plan.v1'&&m.budget===256);
 const jobs=[];
 for(const j of m.jobs){
  const s=j.spec,{seed,turn,board_hash}=s.source,base=fresh(TheySingEngine,seed,turn),frozen=signature(base),players=s.players;
  assert.equal(hash(frozen),board_hash);assert.deepEqual([...players].sort(),[...base.getState().factions.keys()].filter(p=>p!=='NEUTRAL').sort());assert.equal(s.denominator,2*base.getState().nodes.size);
  const seen=new Set(),cells=[];
  assert(j.plan.cells.length===j.plan.required_cells&&j.plan.required_cells<=m.budget);
  for(const t of j.plan.cells){
   const k=t.join(',');assert(t.length===players.length&&!seen.has(k));seen.add(k);
   const e=fresh(TheySingEngine,seed,turn);assert.deepEqual(signature(e),frozen);
   const orders={};
   for(let i=0;i<players.length;i++){const p=players[i],a=t[i];assert(Number.isInteger(a)&&a>=0&&a<s.rosters[p].length);orders[p]=s.rosters[p][a].orders}
   for(const p of [...players].sort())if(orders[p].length){const r=e.submitOrders(p,structuredClone(orders[p]));assert(r.success,p+' '+r.message)}
   e.advancePhase();assert.equal(e.getCurrentPhase(),'RESOLUTION');
   cells.push({actions:t,payoffs:players.map(p=>utility(e,p).numerator),post_state_hash:hash(signature(e))});
  }
  assert.deepEqual(signature(base),frozen);jobs.push({id:s.id,digest:j.digest,board_hash,cells});
 }
 fs.writeFileSync(process.argv[4],JSON.stringify({schema:'kite.native-joint-receipts.v1',jobs},null,2)+'\n');
 console.log('JOINT_THEYSING_CELLS '+jobs.reduce((s,j)=>s+j.cells.length,0));
}
if(require.main===module)main();
