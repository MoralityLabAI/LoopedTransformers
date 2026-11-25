// Retrospective independent JS implementation of LBF15/18 symbolic selectors.
// Use pinned LBF19 cohort, compare geom counts to actual archived native episodes.
import fs from "node:fs";
const cases = fs.readFileSync(process.argv[2], "utf8").replace(/^\uFEFF/, "").trim().split(/\r?\n/).map(JSON.parse);
const cmp = (a, b) => {
  for (let i=0;i<a.length;i++) {
    if (a[i]<b[i]) return -1;
    if (a[i]>b[i]) return 1;
  }
  return 0;
};
function run(c, policy) {
  const t0=Date.now(), L=c.source.length, budget=512, depth=24;
  let f=[[c.source],[c.target]], d=[new Map([[c.source,0]]),new Map([[c.target,0]])],
      par=[new Map([[c.source,null]]),new Map([[c.target,null]])],
      work=0,expanded=0,decisions=0,meet=null,calls=0;
  const mv=c.rules.flatMap(([a,b],i)=>[[i,0,a,b],[i,1,b,a]]);
  const adj=new Map();
  for(const [a,b] of c.rules){
    if(!adj.has(a))adj.set(a,[]);
    if(!adj.has(b))adj.set(b,[]);
    adj.get(a).push(b);adj.get(b).push(a);
  }
  const pd=new Map();
  for(const a of adj.keys()){
    let q=[a],dist=new Map([[a,0]]);
    for(let i=0;i<q.length;i++)for(const b of adj.get(q[i]))if(!dist.has(b)){
      dist.set(b,dist.get(q[i])+1);q.push(b);
    }
    pd.set(a,dist);
  }
  const deg=new Map();
  function degree(w){
    if(deg.has(w))return deg.get(w);
    let k=0;
    for(const [a,b] of c.rules)for(let p=0;p<=L-a.length;p++){
      const sub=w.slice(p,p+a.length);
      if(sub===a||sub===b)k++;
    }
    deg.set(w,k);return k;
  }
  const pc=new Map();
  function features(u,v){
    let key=u+"|"+v,old=pc.get(key);
    if(old)return old;
    let dh=0,first=-1,last=-1;
    for(let p=0;p<L;p++)if(u[p]!==v[p]){
      dh++;if(first<0)first=p;last=p;
    }
    let span=first<0?0:last-first+1,cond=0;
    for(const k of [2,3])for(let p=0;p<=L-k;p++){
      let a=u.slice(p,p+k),b=v.slice(p,p+k);
      if(a!==b){let v0=pd.get(a)?.get(b);if(v0!==undefined)cond+=1/v0;}
    }
    old=[dh,span,cond,Math.ceil(dh/3)];pc.set(key,old);return old;
  }
  while(work<budget&&f[0].length&&f[1].length&&meet===null){
    decisions++;let key=null,act=[0,0];
    for(let s=0;s<2;s++){
      let ms=f[s].length,opp=f[1-s];
      for(let i=0;i<ms;i++){
        let w=f[s][i],g=d[s].get(w),best=policy==="geom_mm"?[999,999,999]:-1e18;
        for(const v of opp){
          let gv=d[1-s].get(v),[dh,span,cond,lb]=s===0?features(w,v):features(v,w);
          if(g+gv+Math.max(1,lb)>depth)continue;
          if(policy==="geom_mm"){
            let cand=[g+gv+Math.max(1,lb),dh,span];if(cmp(cand,best)<0)best=cand;
          }else{
            let sc=-(g+gv+lb)-dh/L+cond/Math.sqrt(1+degree(w)+degree(v));
            if(sc>best)best=sc;
          }
        }
        let cand=policy==="geom_mm"?[...best,ms,g,s,i]:
          [-(best> -1e17?best-Math.log1p(ms)/Math.log1p(budget):-1e18),ms,s,i];
        if(key===null||cmp(cand,key)<0){key=cand;act=[s,i];}
      }
    }
    let [s,i]=act,w=f[s][i],g=d[s].get(w);
    if(g>=depth){f[s].splice(i,1);continue;}
    expanded++;let novel=[];
    outer:for(const [ri,di,a,b] of mv){
      for(let p=w.indexOf(a);p>=0;p=w.indexOf(a,p+1)){
        if(work>=budget)break outer;
        work++;let nxt=w.slice(0,p)+b+w.slice(p+a.length);
        if(d[s].has(nxt))continue;
        d[s].set(nxt,g+1);par[s].set(nxt,[w,ri,di,p]);calls++;
        if(d[1-s].has(nxt)&&g+1+d[1-s].get(nxt)<=depth){meet=nxt;break outer;}
        if(g+1<depth)novel.push(nxt);
      }
    }
    if(meet===null)f[s].splice(i,1,...novel);
  }
  let plen=null;
  if(meet!==null){
    function chain(s){
      let w=meet,a=[];
      while(par[s].get(w)!==null){let [pr,i,d,p]=par[s].get(w);a.push([i,d,p]);w=pr;}
      return a;
    }
    let path=chain(0).reverse().concat(chain(1).map(([i,d,p])=>[i,1-d,p])),w=c.source;
    for(let [i,d,p] of path){
      let [l,v]=c.rules[i];if(d)[l,v]=[v,l];
      if(w.slice(p,p+l.length)!==l)throw Error("path mismatch");
      w=w.slice(0,p)+v+w.slice(p+l.length);
    }
    if(w!==c.target)throw Error("target mismatch");
    plen=path.length;
  }
  return {root:c.root,L,closed:meet!==null,work,expanded,novel_calls:calls,
          decisions,proof_steps:plen,seconds:(Date.now()-t0)/1000};
}
const out={};
for(const pol of ["geom_mm","graph_bridge"]){
  const rows=cases.map(c=>run(c,pol));
  const mean=(k,rs=rows)=>rs.reduce((s,r)=>s+r[k],0)/rs.length;
  out[pol]={closed:rows.filter(r=>r.closed).length,mean_calls:mean("novel_calls"),
    mean_work:mean("work"),mean_seconds:mean("seconds"),
    by_length:[16,20,24].map(L=>({L,closed:rows.filter(r=>r.L===L&&r.closed).length,
      mean_calls:mean("novel_calls",rows.filter(r=>r.L===L))})),rows};
}
console.log(JSON.stringify(out,null,2));
