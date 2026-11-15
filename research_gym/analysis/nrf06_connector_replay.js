/* NRF-06: deterministic standalone JS float64 replay of frozen signed-permutation task.

   Tool-independent computation: call the returned object's evaluate(base64,
   metadata). The caller checks Git blob SHA against the previously SHA-256
   verified checkpoint ledger before passing the bytes. Only input folds 1,
   5 and 6 are used; no model training or changing registered parameters.

   Interface deliberately returns a pure object and never reads a network,
   the filesystem, or the host's environment.
*/
(() => {
  "use strict";
  const N=128, K=8;
  const PERM={101:[0,81,111,24,47,19,46,87],
              103:[39,20,32,47,90,127,59,124],
              107:[78,104,93,2,38,102,46,22]};
  const SIGNS={101:[-1,-1,1,-1,-1,1,-1,-1],
               103:[1,-1,1,-1,-1,1,-1,1],
               107:[-1,1,1,1,1,1,-1,1]};
  function fromBase64(s) {
    const chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const table=new Int16Array(128).fill(-1);
    for(let i=0;i<64;i++)table[chars.charCodeAt(i)]=i;
    const cleaned=s.replace(/\s/g,""),out=new Uint8Array(Math.floor(cleaned.length*3/4));
    let written=0,bits=0,value=0;
    for(let i=0;i<cleaned.length;i++){
      const ascii=cleaned.charCodeAt(i),digit=ascii<128?table[ascii]:-1;
      if(digit<0)continue;
      value=(value<<6)|digit;bits+=6;
      if(bits>=8){bits-=8;out[written++]=(value>>bits)&255;}
    }
    return out.subarray(0,written);
  }
  function weightsFromZip(bytes) {
    const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
    const u16=i=>view.getUint16(i,true),u32=i=>view.getUint32(i,true);
    let end=-1;
    for(let i=bytes.length-22;i>=Math.max(0,bytes.length-65557);i--)
      if(u32(i)===0x06054b50){end=i;break;}
    if(end<0)throw Error("invalid torch ZIP");
    let off=u32(end+16),count=u16(end+10),files={};
    for(let k=0;k<count;k++){
      const n=u16(off+28),extra=u16(off+30),comment=u16(off+32),loc=u32(off+42),len=u32(off+24);
      const name=String.fromCharCode(...bytes.subarray(off+46,off+46+n)).split("/").slice(-2).join("/");
      if(/^data\/[0-3]$/.test(name)){
        if(u16(off+10)!==0)throw Error("compressed storage unsupported");
        const start=loc+30+u16(loc+26)+u16(loc+28);
        files[name]=new Float32Array(bytes.buffer,bytes.byteOffset+start,len/4);
      }
      off+=46+n+extra+comment;
    }
    for(const key of ["data/0","data/2"])if(files[key]?.length!==N*N)
      throw Error("invalid weight shape");
    for(const key of ["data/1","data/3"])if(files[key]?.length!==N)
      throw Error("invalid bias shape");
    return {w1:files["data/0"],b1:files["data/1"],
            w2:files["data/2"],b2:files["data/3"]};
  }
  function mv(w,x,tr=false){
    const y=new Float64Array(N);
    if(!tr)for(let i=0;i<N;i++){
      let s=0,o=i*N;for(let j=0;j<N;j++)s+=w[o+j]*x[j];y[i]=s;
    }else for(let i=0;i<N;i++){
      let s=0;for(let j=0;j<N;j++)s+=w[j*N+i]*x[j];y[i]=s;
    }
    return y;
  }
  function normals(seed){
    let state=seed>>>0;
    return ()=>{
      state=(Math.imul(1664525,state)+1013904223)>>>0;
      const u=(state+.5)/4294967296;
      state=(Math.imul(1664525,state)+1013904223)>>>0;
      const v=(state+.5)/4294967296;
      return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
    };
  }
  function raw(task,fold,n){
    if(!PERM[task]||![1,5,6].includes(fold))throw Error("unregistered task/fold");
    const g=normals((20261115+7919*task+104729*fold)>>>0);
    const X=[],Y=[];
    for(let i=0;i<n;i++){
      const x=Float64Array.from({length:N},()=>g());
      X.push(x);
      Y.push(Float64Array.from({length:K},(_,j)=>Math.tanh(x[PERM[task][j]]*SIGNS[task][j])));
    }
    return {X,Y};
  }
  function advance(m,x,steps){
    const h=Float64Array.from(x);
    for(let t=0;t<steps;t++){
      const a=mv(m.w1,h);
      for(let j=0;j<N;j++)a[j]=Math.tanh(a[j]+m.b1[j]);
      const z=mv(m.w2,a);
      for(let j=0;j<N;j++)h[j]+=.25*(z[j]+m.b2[j]);
    }
    return h;
  }
  function basis(cols){
    const out=[];
    for(const col of cols){
      const v=Float64Array.from(col);
      for(let pass=0;pass<2;pass++)
        for(const u of out){
          let d=0;for(let j=0;j<N;j++)d+=v[j]*u[j];
          for(let j=0;j<N;j++)v[j]-=d*u[j];
        }
      let len=0;for(let j=0;j<N;j++)len+=v[j]*v[j];
      len=Math.sqrt(len);
      if(len<1e-10)throw Error("rank-deficient projection");
      for(let j=0;j<N;j++)v[j]/=len;
      out.push(v);
    }
    return out;
  }
  function means(train){
    const mh=new Float64Array(N),my=new Float64Array(K);
    for(let i=0;i<train.H.length;i++){
      const h=train.H[i],y=train.Y[i];
      for(let j=0;j<N;j++)mh[j]+=h[j];
      for(let j=0;j<K;j++)my[j]+=y[j];
    }
    for(let j=0;j<N;j++)mh[j]/=train.H.length;
    for(let j=0;j<K;j++)my[j]/=train.H.length;
    return {mh,my};
  }
  function supervision(train,mh,my,shift=0){
    const cols=Array.from({length:K},()=>new Float64Array(N));
    for(let i=0;i<train.H.length;i++){
      const h=train.H[i],y=train.Y[(i+shift)%train.Y.length];
      for(let k=0;k<K;k++){
        const c=cols[k],dy=y[k]-my[k];
        for(let j=0;j<N;j++)c[j]+=(h[j]-mh[j])*dy;
      }
    }
    return basis(cols);
  }
  function randomBasis(seed){
    const g=normals(seed);
    return basis(Array.from({length:K},()=>Float64Array.from({length:N},()=>g())));
  }
  function coordinates(){
    return Array.from({length:K},(_,k)=>Float64Array.from({length:N},(_,j)=>j===k?1:0));
  }
  function features(h,U,mh){
    const z=new Float64Array(K);
    for(let k=0;k<K;k++){
      let s=0,u=U[k];
      for(let j=0;j<N;j++)s+=u[j]*(h[j]-mh[j]);
      z[k]=s;
    }
    return z;
  }
  function solve(A,B){
    const a=Array.from({length:K},(_,i)=>Float64Array.from([...A[i],...B[i]]));
    for(let k=0;k<K;k++){
      let best=k;
      for(let i=k+1;i<K;i++)if(Math.abs(a[i][k])>Math.abs(a[best][k]))best=i;
      [a[k],a[best]]=[a[best],a[k]];
      const d=a[k][k];if(Math.abs(d)<1e-13)throw Error("singular ridge");
      for(let j=k;j<2*K;j++)a[k][j]/=d;
      for(let i=0;i<K;i++)if(i!==k){
        const p=a[i][k];
        for(let j=k;j<2*K;j++)a[i][j]-=p*a[k][j];
      }
    }
    return a.map(x=>Array.from(x.slice(K)));
  }
  function fit(train,U,mh,my){
    const A=Array.from({length:K},(_,i)=>Float64Array.from({length:K},(_,j)=>i===j?1:0));
    const B=Array.from({length:K},()=>new Float64Array(K));
    for(let n=0;n<train.H.length;n++){
      const z=features(train.H[n],U,mh),y=train.Y[n];
      for(let i=0;i<K;i++)for(let j=0;j<K;j++){
        A[i][j]+=z[i]*z[j];B[i][j]+=z[i]*(y[j]-my[j]);
      }
    }
    return solve(A,B);
  }
  function mse(evalData,U,mh,my,coeff){
    let err=0;
    for(let n=0;n<evalData.H.length;n++){
      const z=features(evalData.H[n],U,mh);
      for(let j=0;j<K;j++){
        let prediction=my[j];
        for(let i=0;i<K;i++)prediction+=z[i]*coeff[i][j];
        err+=(prediction-evalData.Y[n][j])**2;
      }
    }
    return err/(K*evalData.H.length);
  }
  function derivatives(m,at,half){
    const h=Float64Array.from(at),ds=[];
    for(let step=0;step<half;step++){
      const a=mv(m.w1,h),d=new Float64Array(N);
      for(let j=0;j<N;j++){
        const v=Math.tanh(a[j]+m.b1[j]);a[j]=v;d[j]=1-v*v;
      }
      ds.push(d);
      const z=mv(m.w2,a);
      for(let j=0;j<N;j++)h[j]+=.25*(z[j]+m.b2[j]);
    }
    return ds;
  }
  function leakageFromDerivatives(m,ds,U){
    let total=0;
    for(const u of U){
      const v=Float64Array.from(u);
      for(let step=ds.length-1;step>=0;step--){
        const z=mv(m.w2,v,true),d=ds[step];
        for(let j=0;j<N;j++)z[j]*=d[j];
        const a=mv(m.w1,z,true);
        for(let j=0;j<N;j++)v[j]+=.25*a[j];
      }
      let sq=0;for(let j=0;j<N;j++)sq+=v[j]*v[j];
      for(const q of U){
        let dp=0;for(let j=0;j<N;j++)dp+=v[j]*q[j];
        sq-=dp*dp;
      }
      total+=Math.max(0,sq);
    }
    return Math.sqrt(total/K);
  }
  function median(values){
    if(values.length!==8)throw Error("need eight random-arm objectives");
    const a=[...values].sort((x,y)=>x-y);
    return (a[3]+a[4])/2;
  }
  function evaluate(base64,meta){
    if(![16,32].includes(meta.rounds)||![101,103,107].includes(meta.seed)||
       ![2048,4096].includes(meta.exposure))throw Error("unregistered checkpoint");
    const model=weightsFromZip(fromBase64(base64));
    const half=meta.rounds/2,task=meta.seed;
    const train=raw(task,1,160),evalData=raw(task,5,96),shadow=raw(task,6,12);
    for(const data of [train,evalData,shadow])
      data.H=data.X.map(x=>advance(model,x,half));
    const {mh,my}=means(train);
    const arms={
      supervised:supervision(train,mh,my),
      label_shifted_train_null:supervision(train,mh,my,1),
      first_eight_coordinates:coordinates(),
    };
    for(let i=0;i<8;i++)
      arms["random_"+i]=randomBasis(400000+31*i+17*task);
    const shadowDerivatives=shadow.H.map(h=>derivatives(model,h,half));
    const result={};
    for(const [name,U] of Object.entries(arms)){
      const head=fit(train,U,mh,my),error=mse(evalData,U,mh,my,head);
      let leakage=0;
      for(const ds of shadowDerivatives)
        leakage+=leakageFromDerivatives(model,ds,U);
      leakage/=shadow.H.length;
      result[name]={heldout_mse:error,local_fiber_leakage:leakage,
                    lambda_one_objective:error+leakage**2};
    }
    const randomMedian=median(Array.from({length:8},(_,i)=>result["random_"+i].lambda_one_objective));
    const s=result.supervised.lambda_one_objective;
    return {
      id:"R"+meta.rounds+"_S"+task+"_E"+meta.exposure,
      rounds:meta.rounds,seed:task,exposure:meta.exposure,
      checkpoint_sha256:meta.checkpoint_sha256,git_blob_sha:meta.git_blob_sha,
      arms:result,supervised_objective:s,
      random_median_objective:randomMedian,
      supervised_wins:s<randomMedian,
    };
  }
  return {evaluate};
})()
