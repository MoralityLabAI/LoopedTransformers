from train import *
EXT=2400

def run_extend(seed,arm,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out.mkdir(parents=True,exist_ok=True)
    root=Path('results');r=json.loads((root/f'{seed}-{arm}.json').read_text());p=root/r['checkpoint'];assert sha(p.read_bytes())==r['checkpoint_sha256'];m=fresh(4,seed);m.load_state_dict(torch.load(p,weights_only=True));teacher=Model('tied','affine');tp=Path('parents/affine_long')/f'affine-tied-{seed}.pt';assert sha(tp.read_bytes())==r['parent_sha256'];teacher.load_state_dict(torch.load(tp,weights_only=True));teacher.eval();o=opt(m);gen=torch.Generator().manual_seed(970000+seed);sampler=hashlib.sha256();mac=teachmac=0;h=4 if arm=='fixed4' else 8;start=time.monotonic()
    for _ in range(EXT):
        x,y=stream(gen,r['train_contexts'],B,8);sampler.update(x.numpy().tobytes());x=x[:,:h];y=y[:,:h];z,_=m(x);loss=F.binary_cross_entropy_with_logits(z,y);mac+=B*h*cost(4)
        if arm=='kd_long8':
            with torch.no_grad():soft=(teacher(x)[0]/2).sigmoid();teachmac+=B*h*cost(64)
            loss=.5*loss+2*F.binary_cross_entropy_with_logits(z/2,soft)
        o.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
    cp=out/p.name;torch.save(m.state_dict(),cp);data={t:eval_data(seed,r['dev_contexts'],t) for t in HORIZONS};m.eval();res=dict(seed=seed,arm=arm,steps=r['steps']+EXT,width=4,params=sum(p.numel() for p in m.parameters()),checkpoint=cp.name,checkpoint_sha256=sha(cp.read_bytes()),base_sha256=r['checkpoint_sha256'],parent_sha256=r['parent_sha256'],sampler_sha256=sampler.hexdigest(),train_contexts=r['train_contexts'],cal_contexts=r['cal_contexts'],dev_contexts=r['dev_contexts'],student_mac_proxy=r['student_mac_proxy']+mac,teacher_mac_proxy=r['teacher_mac_proxy']+teachmac,selection_forward_mac_proxy=r['selection_forward_mac_proxy'],eval=measure(m,data),removed=evaluate(m,data[4],remove=True),quant={str(b):measure(quant(m,b),{t:data[t] for t in (8,12)}) for b in (8,4)},seconds=time.monotonic()-start)
    (out/f'{seed}-{arm}.json').write_text(json.dumps(res,separators=(',',':'))+'\n');print(seed,arm,res['steps'],[(t,round(res['eval'][str(t)]['acc']*100,2)) for t in HORIZONS],round(res['seconds'],2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--out',type=Path,default=Path('extended'));a=ap.parse_args();run_extend(a.seed,a.arm,a.out)
