import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {score,repair,reference,encode,windowCounts,mix} from '../kite01.mjs';
// Load checker functions without invoking its CLI or replacing any arithmetic.
const src=readFileSync(new URL('./report.mjs',import.meta.url),'utf8').replace(/^import .*$/gm,'').split('const data=JSON.parse')[0];
const {solveRoot,pureAudit}=Function('assert','score','repair','reference','encode','windowCounts','mix',src+'\nreturn {solveRoot,pureAudit};')(assert,score,repair,reference,encode,windowCounts,mix);
const players=Array.from({length:7},(_,i)=>'p'+i),hash='a'.repeat(64);
const pure={id:'mock-only',players,mode:'hold',denominator:68,rosters:Object.fromEntries(players.map(p=>[p,[{id:'hold'},{id:'move'}]])),anchor:Object.fromEntries(players.map(p=>[p,'hold'])),incumbent:Object.fromEntries(players.map(p=>[p,5])),deviations:Object.fromEntries(players.map((p,i)=>[p,[{id:'hold',numerator:5,post_state_hash:hash},{id:'move',numerator:5+(i===6?1:0),post_state_hash:hash}]])),native_adjudications:8};
const q=pureAudit(pure);assert(q.restricted_le_020&&!q.restricted_pure_nash);assert.equal(q.max_gain_numerator,'1');assert.deepEqual(q.players.filter(x=>x.profitable).map(x=>x.player),['p6']);
const bad=structuredClone(pure);bad.deviations.p6.pop();assert.throws(()=>pureAudit(bad));const lie=structuredClone(pure);lie.deviations.p0[0].numerator=6;assert.throws(()=>pureAudit(lie));
const R=[[2,2],[0,0]],C=[[1,1],[1,1]];
const root={schema:'kite.conditional-bimatrix.v1',id:'mock',actors:['p0','p1'],actions:{row:[{id:'a',orders:[]},{id:'b',orders:[]}],col:[{id:'c',orders:[]},{id:'d',orders:[]}]},denominator:68,source:{campaign_id:'mock-campaign',board_hash:hash},worlds:[{id:'one',R,C},{id:'two',R:[[0,0],[2,2]],C}].map(w=>({...w,outcomes:R.flatMap((r,i)=>r.map((_,j)=>({row:['a','b'][i],col:['c','d'][j],row_utility:{numerator:w.R[i][j]},col_utility:{numerator:w.C[i][j]},post_state_hash:hash})))}))};
const global=solveRoot(root,'global','subjective'),local=solveRoot(root,'local','subjective'),common=solveRoot(root,'local','common');
assert.equal(global.rows.length,15);assert.equal(local.rows[0].regret.max,'1/4');assert.equal(global.rows[0].regret.max,'1/136');assert.equal(common.rows[0].regret.max,'0/1');assert.equal(local.rows[0].certificate.regret.max,'1/5');assert(common.rows[0].certificate&&!common.rows[0].all_worlds_certified);
const corrupted=structuredClone(root);corrupted.worlds[0].outcomes[0].row_utility.numerator=0;assert.throws(()=>solveRoot(corrupted,'global','common'));
console.log(JSON.stringify({status:'PASS',scope:'synthetic receipt checker only',players:7,omitted_deviation_rejected:true,incumbent_mismatch_rejected:true,cell_mismatch_rejected:true,global_regret:global.rows[0].regret.max,local_regret:local.rows[0].regret.max,common_belief_world_failure:true}));
