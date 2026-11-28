function assert(ok,msg){if(!ok)throw Error(msg)}

function sum(a){return a.reduce((s,v)=>s+v,0n)}

function gcd(a,b){while(b){[a,b]=[b,a%b]}return a}

function reduce(a){const d=a.reduce(gcd,0n);return a.map(v=>v/d)}

function valid(g,x,y){const {R,C,D}=g;assert(typeof D==="bigint"&&D>0n,"denominator");assert(x.length>0&&y.length>0&&R.length===x.length&&C.length===x.length,"shape");for(const M of [R,C])for(const row of M){assert(row.length===y.length,"shape");for(const v of row)assert(typeof v==="bigint"&&v>=0n&&v<=D,"payoff grid")}for(const a of [x,y]){assert(a.every(v=>typeof v==="bigint"&&v>=0n)&&sum(a)>0n,"strategy")}}

function argmax(a){let i=0;for(let j=1;j<a.length;j++)if(a[j]>a[i])i=j;return i}

function score(g,x,y){valid(g,x,y);const {R,C,D}=g, sx=sum(x),sy=sum(y);const rv=R.map(row=>sum(row.map((v,j)=>v*y[j]))),cv=y.map((_,j)=>sum(x.map((v,i)=>v*C[i][j])));const i=argmax(rv),j=argmax(cv);const a=rv[i]*sx-sum(rv.map((v,k)=>v*x[k])),b=cv[j]*sy-sum(cv.map((v,k)=>v*y[k])),d=D*sx*sy;return {a,b,d,i,j}}

function inKite(s){return 4n*s.a+s.b<=s.d&&s.a+4n*s.b<=s.d}

function accepted(s){return 5n*s.a<=s.d&&5n*s.b<=s.d}

function repair(g,x,y,s=score(g,x,y)){if(!inKite(s))return null;let xx=x,yy=y;if(5n*s.a>s.d){xx=reduce(x.map((v,k)=>s.d*v+(k===s.i?(5n*s.a-s.d)*sum(x):0n)))}else if(5n*s.b>s.d){yy=reduce(y.map((v,k)=>s.d*v+(k===s.j?(5n*s.b-s.d)*sum(y):0n)))}const out=score(g,xx,yy);assert(accepted(out),"kite repair failure");return {x:xx,y:yy,s:out}}

function fraction(a,b){const d=gcd(a,b);return `${a/d}/${b/d}`}

function encode(s){return {row:fraction(s.a,s.d),col:fraction(s.b,s.d),max:fraction(s.a>s.b?s.a:s.b,s.d),accepted:accepted(s),kite:inKite(s)}}

function reference(g,x,y){const sx=sum(x),sy=sum(y);let a=0n,b=0n;for(let k=0;k<x.length;k++){let z=0n;for(let i=0;i<x.length;i++)for(let j=0;j<y.length;j++)z+=x[i]*y[j]*(g.R[k][j]-g.R[i][j]);if(z>a)a=z}for(let k=0;k<y.length;k++){let z=0n;for(let i=0;i<x.length;i++)for(let j=0;j<y.length;j++)z+=x[i]*y[j]*(g.C[i][k]-g.C[i][j]);if(z>b)b=z}return {a,b,d:g.D*sx*sy}}

function audit(){let total=0,kite=0,row=0,col=0,already=0,outside=0;for(let bits=0;bits<256;bits++){const mat=off=>Array.from({length:2},(_,i)=>Array.from({length:2},(_,j)=>BigInt((bits>>(off+i*2+j))&1)));const g={R:mat(0),C:mat(4),D:1n};for(let i=0;i<=4;i++)for(let j=0;j<=4;j++){const x=[BigInt(i),BigInt(4-i)],y=[BigInt(j),BigInt(4-j)],s=score(g,x,y),q=reference(g,x,y);assert(s.a===q.a&&s.b===q.b&&s.d===q.d,"independent deviation oracle");total++;const p=repair(g,x,y,s);if(!inKite(s)){outside++;assert(p===null,"outside accepted as kite");continue}kite++;assert(p!==null&&accepted(p.s),"repair");const t=reference(g,p.x,p.y);assert(accepted(t),"reference repair");if(5n*s.a>s.d){row++;assert(5n*t.a===t.d,"row boundary")}else if(5n*s.b>s.d){col++;assert(5n*t.b===t.d,"column boundary")}else already++}}let reject=0;for(const [g,x,y] of [[{D:1n,R:[[2n]],C:[[0n]]},[1n],[1n]],[{D:1n,R:[[0n]],C:[[0n]]},[0n],[1n]],[{D:1n,R:[[0n]],C:[[0n]]},[1n],[-1n]]]){try{score(g,x,y)}catch(e){reject++}}assert(reject===3,"input rejection");assert(total===6400&&row>0&&col>0&&outside>0,"coverage");return {total,kite,row_repaired:row,col_repaired:col,already_accepted:already,outside,malformed_rejected:reject}}

function rng(seed){let s=seed>>>0;return n=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return Math.floor(s*n/4294967296)}}

function game(f,n,seed){const random=rng(seed),D=100n;const matrix=fn=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>BigInt(fn(i,j))));let R,C;if(f==="iid"){R=matrix(()=>random(101));C=matrix(()=>random(101))}else if(f==="zero_sum"){R=matrix(()=>random(101));C=R.map(r=>r.map(v=>D-v))}else if(f==="coordination"){R=matrix((i,j)=>i===j?70+random(31):random(41));C=matrix((i,j)=>i===j?70+random(31):random(41))}else if(f==="cyclic"){R=matrix((i,j)=>((i-j+n)%n===1?65:0)+random(36));C=matrix((i,j)=>((j-i+n)%n===1?65:0)+random(36))}else throw Error("family");return {R,C,D}}

function mix(vectors){const totals=vectors.map(sum),product=totals.reduce((s,v)=>s*v,1n);return reduce(vectors[0].map((_,i)=>sum(vectors.map((a,k)=>a[i]*(product/totals[k])))))}

function windowCounts(h,n,w){const a=Array(n).fill(0n);for(const i of h.slice(-w))a[i]++;return a}

function trajectory(g,arm,depths){const n=g.R.length,m=g.R[0].length;let x=Array(n).fill(1n),y=Array(m).fill(1n),hx=[],hy=[],out=[];let solverQueries=0,repairQueries=0;for(let t=0;t<=Math.max(...depths);t++){let xx=x,yy=y;if(arm==="multiscale"&&t>0){xx=mix([x,...[4,16,64].map(w=>windowCounts(hx,n,w))]);yy=mix([y,...[4,16,64].map(w=>windowCounts(hy,m,w))])}const s=score(g,xx,yy);solverQueries+=2;if(depths.includes(t)){const p=repair(g,xx,yy,s);if(p)repairQueries+=2;out.push({depth:t,raw:encode(s),repaired:p?encode(p.s):null,profile:{x:xx.map(String),y:yy.map(String)},certificate:p?{x:p.x.map(String),y:p.y.map(String)}:null,solver_queries:solverQueries,repair_queries:repairQueries})}if(t<Math.max(...depths)){hx.push(s.i);hy.push(s.j);x[s.i]++;y[s.j]++}}return out}

function pilot(){const depths=[0,8,16,32,64],families=["iid","zero_sum","coordination","cyclic"],arms=["fictitious","multiscale"];const rows=[];for(let f=0;f<families.length;f++)for(const n of [4,8])for(let k=0;k<16;k++){const seed=9137+100000*f+1000*n+k*7919;const g=game(families[f],n,seed),root=`${families[f]}-${n}-${k}`;for(const arm of arms)for(const t of trajectory(g,arm,depths))rows.push({root,family:families[f],n,seed,arm,...t})}const summary=[];for(const family of ["all",...families])for(const depth of depths)for(const arm of arms){const v=rows.filter(r=>(family==="all"||r.family===family)&&r.depth===depth&&r.arm===arm);summary.push({family,depth,arm,roots:v.length,raw:v.filter(r=>r.raw.accepted).length,with_repair:v.filter(r=>r.repaired!==null).length,solver_queries:v[0].solver_queries,mean_repair_queries:v.reduce((s,r)=>s+r.repair_queries,0)/v.length})}const paired=[];for(const depth of depths){const a=rows.filter(r=>r.depth===depth&&r.arm==="fictitious"),b=new Map(rows.filter(r=>r.depth===depth&&r.arm==="multiscale").map(r=>[r.root,r]));let wins=0,losses=0,ties=0;for(const r of a){const d=Number(b.get(r.root).repaired!==null)-Number(r.repaired!==null);if(d>0)wins++;else if(d<0)losses++;else ties++}paired.push({depth,wins,losses,ties})}return {status:"development_only",unit:"generated game root",roots:128,depths,arms,rows,summary,paired}}

function perturbAudit(){let checks=0;for(let seed=1;seed<=64;seed++){const g=game("iid",4,seed),r=rng(seed+347),h={D:g.D,R:g.R.map(row=>row.map(v=>{const z=v+BigInt(r(7)-3);return z<0n?0n:z>100n?100n:z})),C:g.C.map(row=>row.map(v=>{const z=v+BigInt(r(7)-3);return z<0n?0n:z>100n?100n:z}))};const x=Array.from({length:4},()=>BigInt(1+r(7))),y=Array.from({length:4},()=>BigInt(1+r(7)));const a=score(g,x,y),b=score(h,x,y);assert(a.d===b.d,"same denominator");for(const [v,w] of [[a.a,b.a],[a.b,b.b],[a.a>a.b?a.a:a.b,b.a>b.b?b.a:b.b]]){const delta=v>w?v-w:w-v;assert(100n*delta<=6n*a.d,"payoff stability");checks++}}return checks}

export {score, encode, inKite, accepted, repair, reference, game, trajectory, audit, perturbAudit, pilot, windowCounts, mix, rng, assert};
