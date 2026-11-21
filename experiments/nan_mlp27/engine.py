"""New MLP workspace and padded reached-state capacity admission, same tasks."""
import gc
from experiments.nan_full26.worker import Engine as OriginalEngine
from .common import write

class Engine(OriginalEngine):
    def __init__(self,a,cfg,out):
        self.capture_attempts=0
        super().__init__(a,cfg,out)
        from mechinterp.nanbeige_bpd_gym.mlp_workspace27 import install
        self.workspace=install(self.loop,cfg['MLP_token_chunk_rows'])

    def capture(self,*args,**kwargs):
        self.capture_attempts+=1
        return super().capture(*args,**kwargs)

    def same(self,a,b):
        t=self.t
        assert a['states'].keys()==b['states'].keys()
        assert all(t.equal(a['states'][k],b['states'][k]) for k in a['states'])
        assert all(t.equal(x,y) for x,y in zip(a['boundaries'],b['boundaries']))
        assert all(t.equal(x,y) for x,y in zip(a['logits'],b['logits']))

    def preflight(self,row):
        t=self.t;ids=self.ids(row);self.embedding.reset();self.workspace.enabled=False
        reference=self.capture([ids],['native']);self.embedding.reset();self.workspace.enabled=True
        chunked=self.capture([ids],['native'])
        changes=[]
        for k,x in reference['states'].items():
            a=x.float();b=chunked['states'][k].float();assert t.isfinite(a).all() and t.isfinite(b).all()
            changes.append(dict(batch_row=k[0],loop=k[1],layer=k[2],site=k[3],
                relative_L2=float((b-a).norm()/a.norm().clamp_min(1e-12)),maximum_abs=float((b-a).abs().max())))
        logits_change=[dict(maximum_abs=float((a-b).abs().max()),argmax_changed=bool(int(a.argmax())!=int(b.argmax())))
            for a,b in zip(reference['logits'],chunked['logits'])]
        assert all(t.isfinite(x).all() for x in chunked['logits'])
        from mechinterp.nanbeige_bpd_gym.resident_head import full_logits
        for h,score in zip(chunked['boundaries'],chunked['logits']):
            assert t.equal(full_logits(self.head.weight,h[:,None,:],self.loop.device,8192),score[0].cpu())
        self.loop.run_pairs=self.old;self.embedding.reset();old_loop=self.capture([ids],['native']);self.same(chunked,old_loop)
        from experiments.nan_full26.batch import install
        install(self.loop);self.embedding.reset();pair=self.capture([ids,ids],['native','native'])
        assert all(t.equal(pair['states'][0,*k[1:]],pair['states'][1,*k[1:]]) for k in pair['states'])
        assert all(t.equal(x[0],x[1]) for x in pair['boundaries']+pair['logits'])
        batch_change=[float((a[0]-b[0]).abs().max()) for a,b in zip(pair['logits'],chunked['logits'])]
        self.install_instrument();self.embedding.reset();all_one=self.capture([ids,ids],['native','all_one']);self.same(pair,all_one)
        del reference,chunked,old_loop,pair,all_one;self.embedding.reset();gc.collect();t.cuda.empty_cache()
        capacity=[]
        for shape in self.cfg['capacity_shapes']:
            B=shape['batch'];P=shape['tokens'];padding=shape['right_padding']
            repeated=(ids*((P+len(ids)-1)//len(ids)))[:P]
            sequences=[repeated] if B==1 else [repeated,repeated[:P-padding]]
            modes=['reached_state'] if B==1 else ['native','reached_state']
            status='passed';failure=None;elapsed=None
            try:
                value=self.capture(sequences,modes,observe=False)
                assert all(t.isfinite(v).all() for v in value['logits'])
                elapsed=value['cuda_elapsed_ms'];del value
            except t.cuda.OutOfMemoryError as error:
                status='capacity_failed';failure=dict(type=type(error).__name__,message=str(error))
            finally:
                self.embedding.reset();gc.collect();t.cuda.empty_cache()
            capacity.append(dict(**shape,status=status,failure=failure,cuda_elapsed_ms=elapsed,
                includes_reached_state_route=True,peak_allocated_bytes=t.cuda.max_memory_allocated(),
                peak_reserved_bytes=t.cuda.max_memory_reserved(),length=P))
        self.capacity=capacity
        admitted=all(self.admitted(s['batch'],s['tokens']) for s in self.cfg['required_capacity'])
        write(self.out/'preflight.json',dict(status='passed' if admitted else 'capacity_not_admitted',
            MLP_chunk_vs_original_all40=changes,MLP_chunk_vs_original_full_logits=logits_change,
            B1_old_new_loop_bit_exact=True,B1_full_head_parity=True,B2_clean_rows_exact=True,B2_all_one_exact=True,
            B1_B2_maximum_logit_abs_difference=batch_change,capacity=capacity,
            required_capacity=self.cfg['required_capacity'],all_full_roster_inputs_and_short_AB_admitted=admitted,
            numerical_differences_do_not_select_tasks=True,completed_preflight_captures=self.forward_count,
            attempted_preflight_captures=self.capture_attempts))
        if not admitted:raise RuntimeError('registered_padded_capacity_gate_failed_before_scored_inference')

    def admitted(self,count,length):
        # Batch-specific probes prevent a B2 pass from rescuing a failed B1
        # shape, or an unpadded pass from admitting padded edited trajectories.
        return any(p['status']=='passed' and p['batch']==count and p['length']>=length
            and p['includes_reached_state_route'] and (count==1 or p['right_padding']>0) for p in self.capacity)
