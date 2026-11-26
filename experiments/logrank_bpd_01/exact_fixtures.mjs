// LOGRANK-BPD-01: standalone exact fixtures, Node.js, no dependencies.
function check(ok,msg){if(!ok)throw Error(msg)}

function ceilLog2(n){return n<=1?0:Math.ceil(Math.log2(n))}

function gcd(a,b){a=a<0n?-a:a;b=b<0n?-b:b;while(b){[a,b]=[b,a%b]}return a}

function rank(M){const a=M.map(r=>r.map(BigInt));let p=0;for(let j=0;j<a[0].length&&p<a.length;j++){let k=p;while(k<a.length&&a[k][j]===0n)k++;if(k===a.length)continue;[a[p],a[k]]=[a[k],a[p]];for(let i=p+1;i<a.length;i++){if(a[i][j]===0n)continue;const u=a[p][j],v=a[i][j];for(let l=j;l<a[0].length;l++)a[i][l]=u*a[i][l]-v*a[p][l];let g=0n;for(let l=j;l<a[0].length;l++)g=gcd(g,a[i][l]);if(g>1n)for(let l=j;l<a[0].length;l++)a[i][l]/=g;}p++;}return p}

function oneWay(M){const ids=new Map(),codes=[];for(const row of M){const key=row.join("");if(!ids.has(key))ids.set(key,ids.size);codes.push(ids.get(key))}const rows=[...ids.keys()];let errors=0;M.forEach((row,x)=>row.forEach((v,y)=>{if(Number(rows[codes[x]][y])!==v)errors++}));return {distinct_rows:ids.size,bits:ceilLog2(ids.size),errors}}

function members(mask,n){return Array.from({length:n},(_,i)=>i).filter(i=>mask&(1<<i))}

function solve(M){const n=M.length,m=M[0].length;check(n<=8&&m<=8,"exact DP size cap");const memo=new Map();
function visit(a,b){const key=a+","+b;if(memo.has(key))return memo.get(key);const xs=members(a,n),ys=members(b,m);if(xs.every(x=>ys.every(y=>M[x][y]===M[xs[0]][y]))){const leaf={cost:0,row:xs[0]};memo.set(key,leaf);return leaf}
let best={cost:Infinity};for(const [party,mask] of [["A",a],["B",b]]){const anchor=mask&-mask;for(let s=(mask-1)&mask;s;s=(s-1)&mask){if(!(s&anchor))continue;const t=mask^s;if(!t)continue;const left=party==="A"?visit(s,b):visit(a,s);const right=party==="A"?visit(t,b):visit(a,t);const cost=1+Math.max(left.cost,right.cost);if(cost<best.cost)best={cost,party,mask:s,left,right};if(best.cost===1)break;}if(best.cost===1)break;}memo.set(key,best);return best;}
const tree=visit((1<<n)-1,(1<<m)-1);let worst=0,total=0,errors=0;for(let x=0;x<n;x++)for(let y=0;y<m;y++){let node=tree,bits=0;while(node.party){const index=node.party==="A"?x:y;node=node.mask&(1<<index)?node.left:node.right;bits++}if(M[node.row][y]!==M[x][y])errors++;worst=Math.max(worst,bits);total+=bits;}check(errors===0&&worst===tree.cost,"DP replay");return {bits:worst,mean_bits:total/(n*m),errors,states:memo.size,tree}}

function rect(M){const n=M.length,m=M[0].length;check(n<=16,"rectangle cap");let best={area:0};for(let a=1;a<(1<<n);a++){const xs=members(a,n);for(let v=0;v<2;v++){const ys=Array.from({length:m},(_,y)=>y).filter(y=>xs.every(x=>M[x][y]===v));const area=xs.length*ys.length;if(area>best.area)best={area,value:v,rows:xs,cols:ys};}}return {...best,density:best.area/(n*m)}}

function vector(i,k){return Array.from({length:k},(_,j)=>((i>>j)&1)?-1:1)}

function parityFixture(){const V=Array.from({length:16},(_,i)=>vector(i,4));const f=x=>x[0]*x[1];const g=y=>y[1]*y[3];const M=V.map(x=>V.map(y=>(1+f(x)*g(y))/2));const trainA=V.filter(x=>x[2]===1),testA=V.filter(x=>x[2]===-1),trainB=V.filter(y=>y[0]===1),testB=V.filter(y=>y[0]===-1);const singles=Array.from({length:4},(_,i)=>[i]),pairs=singles.flatMap(([i])=>singles.filter(([j])=>i<j).map(([j])=>[i,j]));const feature=(x,ix)=>ix.reduce((a,i)=>a*x[i],1);function fit(xs,label,features){let best=null;for(const ix of features)for(const sign of [1,-1]){const mistakes=xs.filter(x=>sign*feature(x,ix)!==label(x)).length;if(best===null||mistakes<best.mistakes)best={ix,sign,mistakes}}return best}
const arms=[];for(const [name,fa,fb] of [["degree1",singles,singles],["degree2",pairs,pairs],["fixed_wrong_pair",[[0,2]],[[0,1]]]]){const a=fit(trainA,x=>2*M[V.indexOf(x)][0]-1,fa),b=fit(trainB,y=>2*M[0][V.indexOf(y)]-1,fb);let correct=0;for(const x of testA)for(const y of testB){const bit=a.sign*feature(x,a.ix);const answer=(1+bit*b.sign*feature(y,b.ix))/2;if(answer===M[V.indexOf(x)][V.indexOf(y)])correct++}arms.push({name,a,b,correct,total:testA.length*testB.length,bits:1});}
check(arms[1].correct===64&&arms[0].correct===32&&arms[2].correct===32,"parity controls");return {rank:rank(M),one_way:oneWay(M),rectangle:rect(M),arms}}

function run(){const fixtures=[];for(const [name,M] of [["equality4",Array.from({length:4},(_,x)=>Array.from({length:4},(_,y)=>Number(x===y)))],["strict_greater4",Array.from({length:4},(_,x)=>Array.from({length:4},(_,y)=>Number(x>y)))],["inner_product2",Array.from({length:4},(_,x)=>Array.from({length:4},(_,y)=>Number(Boolean(x&y&1)!==Boolean(x&y&2))))],["index3",Array.from({length:8},(_,x)=>Array.from({length:3},(_,y)=>(x>>y)&1))]]){const r=rank(M),o=oneWay(M),d=solve(M);check(ceilLog2(r)<=d.bits&&d.bits<=o.bits,"rank and row bounds");fixtures.push({name,rank:r,one_way:o,exact_interactive:d,rectangle:rect(M)});}
let cases=0;for(let mask=0;mask<512;mask++){const M=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>(mask>>(i*3+j))&1));const r=rank(M),d=solve(M),o=oneWay(M);check(o.errors===0&&ceilLog2(r)<=d.bits&&d.bits<=o.bits,"exhaustive bound");cases++}
let errors=0;for(let x=0;x<16;x++)for(let y=0;y<4;y++){const query=[(y>>1)&1,y&1];const decoded=(query[0]<<1)|query[1];const reply=(x>>decoded)&1;if(reply!==((x>>y)&1))errors++}check(errors===0,"index4 replay");return {evidence:"exact finite fixture and development-only feature fit",output_convention:"Bob alone outputs; communication excludes local computation and final local output",exhaustive_3x3_cases:cases,fixtures,index4:{rank:rank(Array.from({length:16},(_,x)=>Array.from({length:4},(_,y)=>(x>>y)&1))),one_way_bits:4,verified_interactive_upper_bound_bits:3,replayed_pairs:64,errors},parity:parityFixture()}}

console.log(JSON.stringify(run(),null,2));
