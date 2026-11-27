import assert from 'node:assert/strict';
import {makeGame} from './joint_search_pilot.mjs';
import {runBoard,summarize} from './native_adaptive_core.mjs';
const {spec,oracle}=makeGame('directed_cycle',9187213);
spec.source={campaign_id:'mock',board_hash:'a'.repeat(64)};
const r=await runBoard(spec,(_,t)=>oracle(t)),s=summarize([r]);
assert.equal(s.boards,1);assert.equal(s.campaign_clusters,1);assert.equal(s.summary.length,3);
assert(r.results.every(x=>x.unique_cells<=256));
assert(r.results.every(x=>x.shared_observed_range.players.length===7));
let rejected=false;
try{await runBoard(spec,async(mode,t)=>{const a=await oracle(t);if(mode==='hybrid')a[0]={...a[0],post_state_hash:'b'.repeat(64)};return a})}catch(e){rejected=true}
assert(rejected,'cross-arm cell mismatch was accepted');
console.log(JSON.stringify({status:'PASS',scope:'mock native adaptive core',summary:s,changed_cell_rejected:rejected}));
