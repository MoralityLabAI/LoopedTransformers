"""Publish complete fixed-tower context probes and paired physical geometry."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from experiments.nan_fit07.io import read,rows,sha,write
from .context_probes import ROOT,cached,groups,root_macro,describe


def physical_pair(bf,nf,meta):
    values={}
    for layer in bf:
        B=bf[layer]['residual_after_layer'];N=nf[layer]['residual_after_layer']
        bn=np.linalg.norm(B,axis=1);nn=np.linalg.norm(N,axis=1);den=bn*nn
        cos=np.divide(np.sum(B*N,axis=1),den,out=np.full_like(den,np.nan),where=den>0)
        deltaB=B[1::2]-B[0::2];deltaN=N[1::2]-N[0::2]
        assert all(meta[i]['id']==meta[i+1]['id'] and meta[i]['loop']==1 and meta[i+1]['loop']==2 for i in range(0,len(meta),2))
        delta_meta=[r for r in meta if r['loop']==2]
        def summarize(V,M,keys):
            out=[]
            for key,ii in sorted(groups(M,keys).items()):
                metrics={k:float(root_macro(a,ii,M)) for k,a in V.items()}
                out.append(describe(key,ii,M,keys)|{k:v if np.isfinite(v) else None for k,v in metrics.items()})
            return out
        keys=('split','context_budget','position','task','copy_control','loop')
        values[str(layer)]=dict(states=summarize({'physical_relative_l2':np.linalg.norm(N-B,axis=1)/np.maximum(bn,1e-12),'physical_cosine':cos},meta,keys),
            loop_displacements=summarize({'BF16_delta_l2':np.linalg.norm(deltaB,axis=1),'NF4_delta_l2':np.linalg.norm(deltaN,axis=1),
                'delta_relative_l2':np.linalg.norm(deltaN-deltaB,axis=1)/np.maximum(np.linalg.norm(deltaB,axis=1),1e-12)},delta_meta,keys[:-1]),
            undefined_state_cosines=int(np.isnan(cos).sum()))
    return values


def report(runs,out):
    backends={}
    for b in ('nf4','bf16'):
        file=runs/f'probes/{b}_context_aggregate.json'
        if not file.exists():continue
        p=read(file);assert p['status']=='passed' and p['contexts']==291
        assert p['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
        assert p['worker_receipt_sha256']==sha(runs/b/'receipt.json')
        assert p['implementation_sha256']==sha(Path(__file__).with_name('context_probes.py'))
        backends[b]=p
    paired=None
    if len(backends)==2:
        public=rows(ROOT/'fixtures/public.jsonl');source=sha(ROOT/'SOURCE_MANIFEST.json')
        B,bm,_=cached(runs/'bf16',public,source);N,nm,_=cached(runs/'nf4',public,source);assert bm==nm
        paired=physical_pair(B,N,bm)
    result=dict(updated_UTC=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),backends=backends,
        implementation_sha256=sha(Path(__file__)),paired_physical_geometry=paired,
        formal_inferential_comparisons=0,BPD_donor_transfer_attempted=False,sealed_test_accessed=False,
        limits='Observational cached-state development probes on four DEV families; backend feature indices are not subtracted. No causal/equivalence claim. Relative-norm denominator floor 1e-12; undefined cosines remain null.')
    write(out/'nan_ctx08_tower_probes.json',result)
    lines=['# NAN-CTX-08 fixed SAE tower and learned VPD probes','',f"Updated {result['updated_UTC']}",'',
        'These CPU probes apply the already fitted NAN-GYM-05 dictionaries and their random initial controls to every '+
        'completed context at layers 5/11/17/21 and both loops. Each backend requires all 291 contexts, including all '+
        'world pairs, positions, lengths, tasks and copy controls. No refitting or outcome selection is performed. '+
        'The operation family was in the frozen CTX08 protocol; this implementation was written after NF4 behavior '+
        'and partial BF16 behavior were exposed. It is an observational development analysis.','',
        '| Backend | Complete contexts | Fixed dictionaries | Learned VPD T1 asset |', '|---|---:|---:|---|']
    for b,p in backends.items():lines.append(f"| {b.upper()} | 291 | 4, each with initial control | {p['learned_VPD']['status']} |")
    for b in ('nf4','bf16'):
        if b not in backends:lines.append(f'| {b.upper()} | pending | pending | pending |')
    lines+=['','All 128 feature activity fractions, normalized reconstruction errors, physical errors, loop/world '+
        'displacements and initial-control counterparts are retained in the aggregate JSON for every registered group. '+
        'Values are averaged within source root before averaging roots. The four DEV families are the units; '+
        'the hundreds of variants, loops and lengths are repeated measures.','',
        'The learned-factor producer is FIT07 only. Its completed BF16 T1 checkpoint is applied to cached layer-21 '+
        'down_proj inputs, with original SVD and matched-norm Haar controls and every SAE-feature/controller-energy '+
        'root correlation. Loop 2 is outside T1 fitting depth. NF4 has no eligible fitted asset; FIT07 T2 failed. '+
        'RECOMP09 is a separate execution amendment and is excluded from this original producer set.','',
        f"Paired BF16/NF4 physical residual and loop-displacement geometry: {'complete at all four layers' if paired else 'pending both complete backend probes'}. "+
        'Feature-index codes from independently fitted backend dictionaries are never subtracted. High cosine '+
        'does not imply equal displacement or a shared causal mechanism. Residuals precede final loop RMSNorm; '+
        'only down_proj has the immediate residual identity bridge.','',
        'No model inference, online intervention, causal test, equivalence claim or donor transfer is performed. '+
        'Raw states, per-record codes, controller masks and IDs remain local. The original 168-case test stays sealed.','',
        '[Behavior and capacity](NAN_CTX08_PROGRESS.md) · [All aggregate probes](nan_ctx08_tower_probes.json) · '+
        '[Original protocol](../../experiments/nan_ctx08/PROTOCOL.md)','']
    (out/'NAN_CTX08_TOWER_PROBES.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print({'complete_backends':list(backends),'paired_geometry':paired is not None})


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/nan_ctx08');p.add_argument('--out',default='reports/nanbeige_bpd_gym');a=p.parse_args()
    report(Path(a.runs),Path(a.out))


if __name__=='__main__':main()
