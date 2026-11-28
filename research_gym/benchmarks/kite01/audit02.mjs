import {readFileSync} from 'node:fs';
import {reference} from './kite01.mjs';
import {makeGame, roots} from './router02.mjs';

function auditReceipts(named){
  const gameRoots=roots("test"), gm=new Map(gameRoots.map(r=>[r.id,{root:r,game:makeGame(r)}]));
  const depths=[4,8,16,32];
  const counts={files:0,roots:gm.size,raw_profiles:0,repair_certificates:0,exact_regret_components:0,query_receipts:0,history_prefix_checks:0,route_counts:{}};
  function ok(v,m){if(!v)throw Error(m)}
  function eq(a,d,f){const [u,v]=f.split("/").map(BigInt);return a*v===u*d}
  function verify(game,profile,record){
    const x=profile.x.map(BigInt),y=profile.y.map(BigInt);
    ok(x.length===game.R.length&&y.length===game.C[0].length&&x.every(v=>v>=0n)&&y.every(v=>v>=0n)&&x.some(v=>v>0n)&&y.some(v=>v>0n),"invalid strategy");
    const v=reference(game,x,y),z=v.a>v.b?v.a:v.b;
    ok(eq(v.a,v.d,record.row)&&eq(v.b,v.d,record.col)&&eq(z,v.d,record.max),"regret mismatch");
    ok(record.accepted===(5n*z<=v.d)&&record.kite===(4n*v.a+v.b<=v.d&&v.a+4n*v.b<=v.d),"flag mismatch");
    counts.exact_regret_components+=3;
    return {v,z}
  }
  for(const [name,pack] of named){
    counts.files++;ok(pack.rows.length===gm.size*depths.length,"row count");const arm=name.startsWith("learned_seed_")?"learned":name,expectedSeed=name.startsWith("learned_seed_")?Number(name.slice(13)):719;
    ok(pack.model_seed===expectedSeed,"seed");
    const group=new Map(),hist={};
    for(const row of pack.rows){
      const info=gm.get(row.root);ok(info&&row.family===info.root.family&&row.seed===info.root.seed&&row.n===info.root.n&&row.arm===arm&&depths.includes(row.depth),"root metadata");
      const {v,z}=verify(info.game,row.profile,row.raw);counts.raw_profiles++;
      ok(row.raw_le_005===(20n*z<=v.d)&&row.raw_le_010===(10n*z<=v.d),"threshold mismatch");
      ok(Boolean(row.certificate)===row.raw.kite,"repair presence");
      if(row.certificate){const q=verify(info.game,row.certificate,row.certificate.regret);ok(5n*q.z<=q.v.d,"invalid certificate");counts.repair_certificates++}
      ok(row.solver_queries===(row.depth+1)*(arm==="oracle"?10:2),"query count");
      ok(row.actions.length===row.depth+1&&row.actions.every(k=>Number.isInteger(k)&&k>=0&&k<5)&&row.actions.at(-1)===row.route,"route history");counts.query_receipts++;
      if(!group.has(row.root))group.set(row.root,[]);group.get(row.root).push(row);
      if(!hist[row.depth])hist[row.depth]=[0,0,0,0,0];hist[row.depth][row.route]++;
    }
    for(const rr of group.values()){rr.sort((a,b)=>a.depth-b.depth);ok(rr.map(r=>r.depth).join(",")===depths.join(","),"checkpoint coverage");let repairs=0;for(let i=0;i<rr.length;i++){const r=rr[i];if(r.certificate)repairs+=2;ok(r.repair_queries===repairs,"repair query count");if(i){ok(rr[i-1].actions.every((k,j)=>r.actions[j]===k),"history mismatch");counts.history_prefix_checks++}}}
    ok(group.size===gm.size,"root coverage");counts.route_counts[name]=hist;
  }
  return counts
}

const names=["full","w4","w16","w64","uniform","learned","shuffled","oracle","learned_seed_1723","learned_seed_4219"];
console.log(JSON.stringify(auditReceipts(names.map(n=>[n,JSON.parse(readFileSync(new URL('./router02_receipts/'+n+'.json',import.meta.url),'utf8'))])),null,2));
