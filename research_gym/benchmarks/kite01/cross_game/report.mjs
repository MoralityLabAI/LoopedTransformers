import {readFileSync,writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {score,repair,reference,encode,windowCounts,mix} from '../kite01.mjs';
const sum=a=>a.reduce((s,v)=>s+v,0n);
const max=a=>a.reduce((x,y)=>x>y?x:y);
const min=a=>a.reduce((x,y)=>x<y?x:y);
const integer=x=>{assert(typeof x==='string'&&/^\d+$/.test(x)||Number.isSafeInteger(x)&&x>=0,'integer');return BigInt(x)};
const ratio=s=>{const [n,d]=s.split('/').map(Number);return n/d};
function certify(g,x,y){
 const s=score(g,x,y),r=reference(g,x,y);assert(s.a===r.a&&s.b===r.b&&s.d===r.d,'direct regret mismatch');
 const p=repair(g,x,y,s);if(p){const q=reference(g,p.x,p.y);assert(5n*q.a<=q.d&&5n*q.b<=q.d,'bad certificate')}
 return {profile:{x:x.map(String),y:y.map(String)},regret:encode(s),certificate:p?{x:p.x.map(String),y:p.y.map(String),regret:encode(p.s)}:null}
}
function compile(root,scale,belief){
 assert.equal(root.schema,'kite.conditional-bimatrix.v1');assert(root.actors.length===2&&root.actors[0]!==root.actors[1]);
 const m=root.actions.row.length,n=root.actions.col.length,D=integer(root.denominator);
 for(const a of [root.actions.row,root.actions.col])assert(a.length>=2&&new Set(a.map(v=>v.id)).size===a.length);
 assert(root.worlds.length===2&&new Set(root.worlds.map(w=>w.id)).size===2&&D>0n);
 const matrix=M=>{assert(M.length===m);return M.map(r=>{assert(r.length===n);return r.map(v=>{const z=integer(v);assert(z<=D);return z})})};
 let worlds=root.worlds.map(w=>{const R=matrix(w.R),C=matrix(w.C);assert(w.outcomes.length===m*n);for(let i=0;i<m;i++)for(let j=0;j<n;j++){const q=w.outcomes[i*n+j];assert(q.row===root.actions.row[i].id&&q.col===root.actions.col[j].id);assert(integer(q.row_utility.numerator)===R[i][j]&&integer(q.col_utility.numerator)===C[i][j]);assert(/^[a-f0-9]{64}$/.test(q.post_state_hash))}return {id:w.id,R,C,D}});
 const r=worlds.flatMap(w=>w.R.flat()),c=worlds.flatMap(w=>w.C.flat()),loR=min(r),loC=min(c),spanR=max(r)-loR,spanC=max(c)-loC;
 if(scale==='local'){const a=spanR||1n,b=spanC||1n;worlds=worlds.map(w=>({id:w.id,D:a*b,R:w.R.map(row=>row.map(v=>(v-loR)*b)),C:w.C.map(row=>row.map(v=>(v-loC)*a))}))}
 const wr=belief==='subjective'?[3n,1n]:[1n,1n],wc=belief==='subjective'?[1n,3n]:[1n,1n],sr=sum(wr),sc=sum(wc),weighted=(k,w,other)=>Array.from({length:m},(_,i)=>Array.from({length:n},(_,j)=>other*sum(worlds.map((g,k2)=>w[k2]*g[k][i][j]))));
 return {g:{R:weighted('R',wr,sc),C:weighted('C',wc,sr),D:worlds[0].D*sr*sc},worlds,m,n,span:[String(spanR),String(spanC)],originalD:String(D)}
}
function solveRoot(root,scale,belief){
 const {g,worlds,m,n,span,originalD}=compile(root,scale,belief),rows=[];
 const emit=(arm,depth,x,y)=>{const r=certify(g,x,y),p=r.certificate;rows.push({arm,depth,solver_products:2*depth,...r,world_audit:worlds.map(w=>({world:w.id,raw:encode(reference(w,x,y)),repaired:p?encode(reference(w,p.x.map(BigInt),p.y.map(BigInt))):null})),all_worlds_certified:!!p&&worlds.every(w=>{const s=reference(w,p.x.map(BigInt),p.y.map(BigInt));return 5n*s.a<=s.d&&5n*s.b<=s.d})})};
 for(const mode of ['full','multiscale']){const x=Array(m).fill(1n),y=Array(n).fill(1n),hx=[],hy=[];for(let t=0;t<=64;t++){
  const bank=(a,h)=>h.length?mix([a,...[2,4,8,16].map(w=>windowCounts(h,a.length,w))]):a;
  const qx=mode==='full'?x:bank(x,hx),qy=mode==='full'?y:bank(y,hy);
  if([0,8,16,32,64].includes(t)){emit(mode==='full'?'full':'workspace_full_readout',t,x,y);if(mode==='multiscale')emit('workspace_direct_readout',t,qx,qy)}
  if(t<64){const s=score(g,qx,qy);x[s.i]+=BigInt(m);y[s.j]+=BigInt(n);hx.push(s.i);hy.push(s.j)}
 }}
 return {id:root.id,actors:root.actors,source:root.source,scale,belief,span,originalD,rows};
}

function pureAudit(p){
 const D=integer(p.denominator);assert(D>0n&&p.players.length>1&&new Set(p.players).size===p.players.length);
 for(const k of ['rosters','anchor','incumbent','deviations'])assert.deepEqual(Object.keys(p[k]).sort(),[...p.players].sort(),'player coverage');
 const results=[];let n=1;
 for(const a of p.players){
  const ids=p.rosters[a].map(x=>x.id),rows=p.deviations[a];assert(ids.length&&new Set(ids).size===ids.length&&rows.length===ids.length);
  assert.deepEqual(rows.map(x=>x.id).sort(),[...ids].sort(),'deviation coverage');
  const q=integer(p.incumbent[a]);assert(q<=D);const inc=rows.find(v=>v.id===p.anchor[a]);assert(inc&&integer(inc.numerator)===q,'incumbent mismatch');
  for(const r of rows)assert(integer(r.numerator)<=D&&/^[a-f0-9]{64}$/.test(r.post_state_hash),'invalid deviation receipt');
  const best=rows.reduce((x,y)=>integer(y.numerator)>integer(x.numerator)?y:x),gain=integer(best.numerator)-q;assert(gain>=0n);
  results.push({player:a,incumbent:p.anchor[a],best_alternative:best.id,incumbent_payoff:String(q),gain_numerator:String(gain),profitable:gain>0n});
  n+=rows.length-1;
 }
 assert(n===p.native_adjudications,'native adjudication count');
 const z=max(results.map(v=>BigInt(v.gain_numerator)));
 return {id:p.id,players:results,mode:p.mode,max_gain_numerator:String(z),denominator:String(D),max_regret:Number(z)/Number(D),restricted_pure_nash:z===0n,restricted_le_020:5n*z<=D,native_adjudications:n,scope:'all declared unilateral alternatives to this single pure joint profile; fixed other actions and engine seed; no omitted-action or campaign guarantee'};
}

const data=JSON.parse(readFileSync(process.argv[2],'utf8')),out=process.argv[3];assert(data.roots.length>0&&new Set(data.roots.map(r=>r.id)).size===data.roots.length);
const roots=data.roots.flatMap(r=>['global','local'].flatMap(s=>['common','subjective'].map(b=>solveRoot(r,s,b))));
const summary=[];for(const scale of ['global','local'])for(const belief of ['common','subjective'])for(const depth of [0,8,16,32,64])for(const arm of ['full','workspace_full_readout','workspace_direct_readout']){
 const a=roots.filter(r=>r.scale===scale&&r.belief===belief).map(r=>r.rows.find(v=>v.depth===depth&&v.arm===arm));
 summary.push({scale,belief,depth,arm,n:a.length,mean_regret:a.reduce((s,r)=>s+ratio(r.regret.max),0)/a.length,raw_le_020:a.filter(r=>r.regret.accepted).length,certified:a.filter(r=>r.certificate).length,all_worlds:a.filter(r=>r.all_worlds_certified).length})
}
const nativeCells=data.roots.reduce((s,r)=>s+r.worlds.reduce((v,w)=>v+w.outcomes.length,0),0),clusters=new Set(data.roots.map(r=>{assert(typeof r.source.campaign_id==='string'&&r.source.campaign_id.length>0,'missing campaign cluster');return r.source.campaign_id})).size;
const pure=(data.pure_profiles??[]).map(pureAudit);
const report={pure_profiles:pure,pure_adjudications:pure.reduce((z,p)=>z+p.native_adjudications,0),status:'native_development_only',nativeCells,roots:data.roots.length,clusters,root_details:roots.map(r=>({id:r.id,scale:r.scale,belief:r.belief,span:r.span,originalD:r.originalD})),summary,profiles:roots};
writeFileSync(out,JSON.stringify(report,null,2)+'\n');console.log('KITE_NATIVE_SUMMARY '+JSON.stringify({nativeCells,pureAdjudications:report.pure_adjudications,pure:pure.map(p=>({id:p.id,regret:p.max_regret,restricted_pure_nash:p.restricted_pure_nash,profitable:p.players.filter(v=>v.profitable).map(v=>v.player)})),roots:report.roots,clusters,summary:summary.filter(s=>s.depth===0||s.depth===32),spans:report.root_details.filter(r=>r.scale==='global'&&r.belief==='common').map(r=>({id:r.id,span:r.span,D:r.originalD}))}));
