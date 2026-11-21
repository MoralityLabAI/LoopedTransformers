"""CAL-only closure, full-vocabulary head and required cached capacity gates."""
import gc
import time
from .common import write,sha


def save(engine,name,value):
    """Small complete preflight outputs stay private, including all logits."""
    import numpy as np
    t=engine.t;arrays={};index=[]
    for (row,phase,layer,site),vector in value['states'].items():
        key='v'+str(len(index));arrays[key]=vector.view(t.uint16).numpy()
        index.append(dict(row=row,loop=phase,layer=layer,site=site,key=key))
    arrays['boundaries']=t.stack(value['boundaries']).view(t.uint16).numpy()
    arrays['full_logits']=t.stack([x.detach().cpu() for x in value['logits']]).numpy()
    if engine.cache is not None and engine.cache.instrument is not None:
        instrument=engine.cache.instrument
        for field in ('native','inputs','masks'):
            values=getattr(instrument,field)
            if values:
                ordered=sorted(values);data=t.stack([values[key] for key in ordered])
                arrays[field]=data.view(t.uint16).numpy() if data.dtype==t.bfloat16 else data.numpy()
                index.append(dict(instrument_field=field,phase_row_order=[list(key) for key in ordered]))
    path=engine.out/'preflight'/f'{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:np.savez(stream,**arrays)
    engine.preflight_bytes+=path.stat().st_size
    write(path.with_suffix('.json'),dict(name=name,state_file=path.name,state_sha256=sha(path),index=index,
        full_logits_saved=True,decoder_capture=engine.captures,sealed_test_accessed=False))


def same(t,left,right,rows=None):
    assert left['states'].keys()==right['states'].keys()
    for key,value in left['states'].items():
        if rows is None or key[0] in rows:assert t.equal(value,right['states'][key]),key
    for x,y in zip(left['boundaries']+left['logits'],right['boundaries']+right['logits']):
        if rows is None:assert t.equal(x,y)
        else:
            for row in rows:assert t.equal(x[row],y[row])


def original_reference(engine,ids):
    from experiments.nan_full26.batch import BatchObserver,Embeddings
    t=engine.t;observer=BatchObserver(engine.loop,engine.cfg['layers_zero_based']);embedding=Embeddings(engine.loop)
    boundaries=[];scores=[]
    try:
        with t.inference_mode():
            h,ctx=embedding.prepare([ids]);observer.lengths=[len(ids)];observer.enabled=True;observer.values={}
            for phase in (1,2):
                engine.loop.phase=phase;observer.phase=phase;h=engine.loop.run_pairs([(h,ctx)])[0]
                last=h[:,-1];boundaries.append(last.detach().cpu().clone());scores.append(engine.head(last).detach().cpu())
            states={key:value.detach().cpu() for key,value in observer.values.items()}
        engine.captures+=1
        assert len(states)==40
        return dict(states=states,boundaries=boundaries,logits=scores)
    finally:observer.close();embedding.reset()


def audit_write(engine,value):
    import numpy as np
    from experiments.nan_fit07.pointwise import masks
    from experiments.nan_story23.host import local_replay
    t=engine.t;instrument=engine.cache.instrument;asset=instrument.base.asset
    if not hasattr(engine,'preflight_bank'):
        with np.load(asset['checkpoint'],allow_pickle=False) as z:engine.preflight_bank={k:z[k] for k in z.files}
    bank=engine.preflight_bank;checks=[]
    for row,mode in enumerate(engine.cache.modes):
        for phase in (1,2):
            x=value['states'][row,phase,21,'down_proj_input'];y=value['states'][row,phase,21,'down_proj_output']
            assert x.dtype==y.dtype==instrument.inputs[phase,row].dtype==instrument.native[phase,row].dtype==t.bfloat16,'Lossless BF16 trace storage requires BF16 physical and instrument coordinates'
            assert t.equal(x,instrument.inputs[phase,row]),'Instrument/physical input coordinate mismatch'
            if mode in ('native','all_one'):
                assert t.equal(y,instrument.native[phase,row]),'Native/all-one local output changed';continue
            actual=instrument.masks[phase,row].numpy();xx=x.float().numpy().astype(np.float64)
            shadow=masks(xx[None],bank,asset['receipt']['final_checkpoint']['CI_config'])[0]
            expected=shadow if mode=='reached_state' else np.asarray(asset['static_CAL_mask'])[phase-1] if mode=='static_CAL_mean' else np.ones(8)
            np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=1e-5)
            native=instrument.native[phase,row].float().numpy()
            predicted=local_replay(xx,native,actual,bank,asset['receipt']['training_contract']['ablation'],phase-1)
            yy=y.float().numpy();relative=float(np.linalg.norm(predicted-yy)/max(float(np.linalg.norm(yy)),1e-12))
            assert relative<=.03,'Independent preflight BPD local-write replay failed'
            checks.append(dict(row=row,loop=phase,mode=mode,maximum_mask_abs_error=float(np.max(np.abs(actual-expected))),local_write_relative_L2=relative))
    return checks


def run(engine,row):
    assert row['split']=='calibration','Preflight cannot use a development source'
    t=engine.t;ids=engine.ids(row);result=dict(status='started',capacity=[],raw_outputs_saved=True,selection_uses_no_task_correctness=True)
    path=engine.out/'preflight.json'
    try:
        import numpy as np
        head_arrays={};head_exact=[]
        # No K/V is allocated during this temporary full-head comparison.
        with t.inference_mode():
            weight=engine.loop.model.lm_head.weight.to(engine.loop.device)
            for B in (1,2):
                h=t.randn(B,weight.shape[1],device=engine.loop.device,dtype=weight.dtype)
                expected=t.cat([t.nn.functional.linear(h,weight[s:s+8192]).float() for s in range(0,len(weight),8192)],-1)
                actual=engine.head(h);head_exact.append(bool(t.equal(actual,expected)))
                head_arrays[f'B{B}_hidden_BF16_bits']=h.detach().cpu().view(t.uint16).numpy()
                head_arrays[f'B{B}_expected_logits']=expected.detach().cpu().numpy();head_arrays[f'B{B}_actual_logits']=actual.detach().cpu().numpy()
            headfile=engine.out/'preflight/head_parity.npz';headfile.parent.mkdir(parents=True,exist_ok=True)
            with headfile.open('xb') as stream:np.savez(stream,**head_arrays)
            engine.preflight_bytes+=headfile.stat().st_size
            result['head_parity_file']=dict(file='preflight/head_parity.npz',sha256=sha(headfile),B1_exact=head_exact[0],B2_exact=head_exact[1])
            assert all(head_exact),'Streamed full-vocabulary head parity'
            del weight,h,expected,actual,head_arrays
        gc.collect();t.cuda.empty_cache();engine.gpu_check()
        result['head_B1_B2_same_chunk_parity']=True
        reference=original_reference(engine,ids);save(engine,'original_B1',reference)
        engine.reset_runtime(instrumented=False);cached=engine.prefill([ids],['native']);save(engine,'cached_B1',cached)
        differences=[]
        for key,value in reference['states'].items():
            a=value.float();b=cached['states'][key].float();assert t.isfinite(a).all() and t.isfinite(b).all()
            differences.append(dict(row=key[0],loop=key[1],layer=key[2],site=key[3],max_abs=float((a-b).abs().max()),
                relative_L2=float((b-a).norm()/a.norm().clamp_min(1e-12))))
        result['full_prefix_vs_cached_all40']=differences
        result['full_prefix_vs_cached_logits']=[dict(max_abs=float((a.cpu()-b.cpu()).abs().max()),argmax_changed=bool(int(a.argmax())!=int(b.argmax())))
            for a,b in zip(reference['logits'],cached['logits'])]
        pair=engine.prefill([ids,ids],['native','native']);save(engine,'uninstrumented_B2',pair)
        assert all(t.equal(pair['states'][0,*key[1:]],pair['states'][1,*key[1:]]) for key in pair['states'])
        assert all(t.equal(x[0],x[1]) for x in pair['boundaries']+pair['logits'])
        engine.reset_runtime(instrumented=True)
        native=engine.prefill([ids,ids],['native','native']);save(engine,'instrumented_native_B2',native);same(t,pair,native)
        closed=engine.prefill([ids,ids],['native','all_one']);save(engine,'all_one_B2',closed);same(t,native,closed)
        result.update(native_hook_closure=True,duplicate_clean_B2_exact=True,all_one_B2_exact=True)
        for route in engine.cfg['AB_routes']:
            edited=engine.prefill([ids,ids],['native',route]);save(engine,route+'_B2',edited);same(t,native,edited,rows=[0])
            result.setdefault('independent_mask_and_write_checks',[]).extend(audit_write(engine,edited))
        result['all_three_routes_preserve_clean_A']=True
        # Exercise the actual incremental and physical-row-pruning path.
        ragged=engine.prefill([ids,ids[:-1]],['native','reached_state']);save(engine,'ragged_incremental_prefill',ragged);del ragged
        v=engine.increment([ids+[7],ids[:-1]+[7]],[7,7]);save(engine,'incremental_B2',v)
        engine.cache.select_rows([1]);v=engine.increment([ids[:-1]+[7,11]],[11]);save(engine,'pruned_incremental_B1',v)
        result['independent_mask_and_write_checks'].extend(audit_write(engine,v))
        result['increment_and_padded_row_pruning_checked']=True
        del reference,cached,pair,native,closed,edited,v;engine.cache.reset();gc.collect();t.cuda.empty_cache()
        for shape in engine.cfg['required_capacity']:
            B,P,pad=shape['batch'],shape['tokens'],shape['right_padding'];capacity=dict(**shape,status='started',passes=[])
            result['capacity'].append(capacity)
            for offset in (0,1):
                length=P-offset;repeated=(ids*((length+len(ids)-1)//len(ids)))[:length]
                sequences=[repeated] if B==1 else [repeated,repeated[:length-pad]]
                modes=['reached_state'] if B==1 else ['native','reached_state']
                engine.observe_peaks();t.cuda.reset_peak_memory_stats();start=time.monotonic();v=engine.prefill(sequences,modes)
                prefill_ms=v['cuda_elapsed_ms'];increment_ms=None
                if offset:
                    save(engine,f'capacity_B{B}_P{P}_before_decode',v)
                    sequences=[x+[7] for x in sequences];v=engine.increment(sequences,[7]*B)
                    increment_ms=v['cuda_elapsed_ms']
                assert engine.cache.physical_length==P
                save(engine,f'capacity_B{B}_P{P}_'+('decode' if offset else 'prefill'),v)
                result['independent_mask_and_write_checks'].extend(audit_write(engine,v))
                sample=engine.gpu_check()
                capacity['passes'].append(dict(prefill_tokens=length,incremental_step=bool(offset),physical_cache_tokens=P,
                    prefill_cuda_elapsed_ms=prefill_ms,increment_cuda_elapsed_ms=increment_ms,
                    cache_bytes=engine.cache.cache_bytes(),wall_seconds=time.monotonic()-start,GPU_allocation_sample=sample,
                    CUDA_peak_allocated_bytes=t.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=t.cuda.max_memory_reserved()))
                del v;engine.cache.reset();gc.collect();t.cuda.empty_cache()
            capacity['status']='passed'
        result['status']='passed';result['decoder_captures']=engine.captures
        result['pretrained_head_only_random_checks']=2
        return result
    except BaseException as error:
        result.update(status='failed',failure=dict(type=type(error).__name__,message=str(error)),decoder_captures=engine.captures)
        raise
    finally:write(path,result)
