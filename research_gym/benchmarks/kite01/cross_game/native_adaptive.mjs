import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {createRequire} from 'node:module';
import {writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {runBoard,summarize} from './native_adaptive_core.mjs';

const require=createRequire(import.meta.url),dir=path.dirname(fileURLToPath(import.meta.url));
async function diplomacy(root){
 const child=spawn(process.env.PYTHON||'python',[path.join(dir,'oracle_diplomacy.py')],{cwd:path.resolve(root),env:{...process.env,PYTHONPATH:path.resolve(root),PYTHONHASHSEED:'0'},stdio:['pipe','pipe','inherit']});
 const exit=new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',code=>resolve(code))});
 const rl=createInterface({input:child.stdout,crlfDelay:Infinity}),lines=rl[Symbol.asyncIterator]();
 async function request(x){
  assert(child.stdin.writable,'oracle closed');
  child.stdin.write(JSON.stringify(x)+'\n');
  const r=await lines.next();assert(!r.done,'oracle terminated before reply');
  return JSON.parse(r.value);
 }
 try{
  const {specs}=await request({op:'describe'}),out=[];
  assert.equal(specs.length,2);
  for(const spec of specs)out.push(await runBoard(spec,async(mode,tuples)=>(await request({op:'evaluate',board:spec.id,mode,tuples})).rows));
  child.stdin.end();assert.equal(await exit,0,'native oracle exit');return out;
 }finally{rl.close();if(!child.killed&&child.exitCode===null)child.kill()}
}
function singBoards(enginePath){
 const {TheySingEngine}=require(path.resolve(enginePath));
 const {fresh,signature,hash,utility,pool,roster}=require('./export_theysing.cjs');
 const out=[];
 // New seed values, same opening/hold-chain family. Not a sealed family.
 for(const seed of [7311,7319])for(const turn of [1,2]){
  const base=fresh(TheySingEngine,seed,turn),frozen=signature(base),s=base.getState(),players=[...s.factions.keys()].filter(p=>p!=='NEUTRAL').sort();
  assert.equal(players.length,7);
  const targets=new Map(players.map(p=>[p,new Set(pool(base,p).map(v=>v.node))]));
  const shared=new Set(players.flatMap(p=>[...targets.get(p)].filter(t=>players.filter(q=>targets.get(q).has(t)).length>=2)));
  const rosters=Object.fromEntries(players.map(p=>[p,roster(base,p,shared,4)]));
  const spec={schema:'kite.sparse-joint.v1',id:['sing-adaptive',seed,turn].join(':'),players,rosters,denominator:2*s.nodes.size,profiles:[{id:'start',weights:Object.fromEntries(players.map(p=>[p,['1',...rosters[p].slice(1).map(()=> '0')]]))}],source:{board_hash:hash(frozen),campaign_id:'theysing-seed-'+seed,seed,turn,engine:'TheySingEngine'}};
  const used=new Map();
  async function oracle(mode,tuples){
   if(!used.has(mode))used.set(mode,new Set());const seen=used.get(mode);
   assert(seen.size+tuples.length<=256);
   const cells=[];
   for(const t of tuples){
    const k=t.join(',');assert(t.length===players.length&&!seen.has(k));seen.add(k);
    const e=fresh(TheySingEngine,seed,turn);assert.deepEqual(signature(e),frozen);
    for(let i=0;i<players.length;i++){
     const p=players[i],a=t[i];assert(Number.isInteger(a)&&a>=0&&a<rosters[p].length);
     const orders=rosters[p][a].orders;if(orders.length){const r=e.submitOrders(p,structuredClone(orders));assert(r.success,p+' '+r.message)}
    }
    e.advancePhase();assert.equal(e.getCurrentPhase(),'RESOLUTION');
    cells.push({actions:t,payoffs:players.map(p=>utility(e,p).numerator),post_state_hash:hash(signature(e))});
   }
   assert.deepEqual(signature(base),frozen);return cells;
  }
  out.push({spec,oracle});
 }
 return out;
}
const [kind,engine,out]=process.argv.slice(2);
assert(['dip','sing'].includes(kind)&&engine&&out,'usage: node native_adaptive.mjs dip|sing ENGINE OUT');
let boards;
if(kind==='dip')boards=await diplomacy(engine);else{boards=[];for(const b of singBoards(engine))boards.push(await runBoard(b.spec,b.oracle))}
const summary=summarize(boards);
writeFileSync(out,JSON.stringify({summary,boards},null,2)+'\n');
console.log('JOINT03_NATIVE_SUMMARY '+JSON.stringify({...summary,per_board:boards.map(b=>({id:b.spec.id,union_cells:b.union_cells,spans:b.shared_spans,arms:b.results.map(r=>({mode:r.mode,start:r.start,final:r.final,local:r.shared_observed_range.max_regret,calls:r.unique_cells,exact:r.certificate.restricted_exact_nash,stop:r.stop,pruned:r.pruned}))}))}));
