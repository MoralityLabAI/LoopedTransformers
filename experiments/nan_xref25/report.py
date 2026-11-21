"""Separate repaired-audit report; original failed reports stay unchanged."""
import json
from pathlib import Path
from .common import ROOT,RUNS,SOURCES,read,sha


def main():
    for source,module in SOURCES.items():
        backends={};folder=Path('runs')/module.split('.')[-1]
        for backend in ('nf4','bf16'):
            file=RUNS/'audits'/(source+'_'+backend+'.json');value=dict(status='pending_separate_repair',cells=[])
            if file.exists():
                audit=read(file);value=dict(status=audit['status'],repair_audit=audit,repair_audit_sha256=sha(file),cells=[])
                if audit['status']=='passed':
                    receipt=folder/backend/'receipt.json';assert sha(receipt)==audit['upstream_receipt_sha256']
                    value['cells']=read(receipt)['cells']
            backends[backend]=value
        payload=dict(experiment='NAN-XREF-25',source=source,source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),backends=backends,
            orientation_only_failures_preserved=True,orientation_only_XREF22_status='unrun_registered_supervision_amendment',
            original_failures_preserved=True,arithmetic_change='Exactly three decoder transposes and one explicit FP64 direction cast and one explicit FP64 direction cast; original frozen sources and worker outputs unchanged.',
            unit='Reused cached source root/family; features/worlds/loops correlated',scientific_causal_floor=None,matched_sham_floor=None,
            estimated_causal_power='uncalibrated observational host validation',formal_inferential_comparisons=0,new_model_calls=0,new_grades=0,
            verdict='REDESIGN_OR_PILOT_ONLY',sealed_test_accessed=False)
        dest=Path('reports/nanbeige_bpd_gym');stem='nan_xref25_'+source+'_progress'
        (dest/(stem+'.json')).write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
        lines=['# NAN-XREF-25: separate '+source+' auditor repair','',
            'Original frozen worker/auditor sources and original failed attempts are preserved. This separate host-only audit adds the three decoder transposes and one explicit FP64 direction cast and one explicit FP64 direction cast required by the stored[3072,128] layout and producer float64 geometry arithmetic. No model call, fitting or new grade.','',
            '| Backend | Separate repair | Depth | Cell | Vectors checked | Geometry cells checked |','|---|---|---:|---|---:|---:|']
        for backend,value in backends.items():
            if not value['cells']:lines.append(f"| {backend} | {value['status']} | — | pending/missing | — | — |")
            checks={c['passes']:c for c in value.get('repair_audit',{}).get('original_arithmetic_audit',{}).get('cells',[])}
            for cell in value['cells']:
                check=checks.get(cell['passes'],{})
                lines.append(f"| {backend} | {value['status']} | {cell['passes']} | {cell['status']} | {check.get('cached_vectors','unrun')} | {check.get('physical_geometry_cells','unrun')} |")
        lines+=['','All available vectors, all128 features/top4, fixed CAL-only dictionaries, initial/random/SVD/Haar controls, root strata and undefined correlations remain in the complete JSON. This engineering repair does not change an inference criterion, replay a worker, reconstruct a missing grade or turn an old failure into a success.','',
            'XREF19 numerical inputs and original failures were already exposed before this separate repair. XREF22 inputs remain subject to its frozen admission rules. The same unchanged independent reconstruction, controller, root averaging and physical geometry checks must pass after the mechanically reversible orientation fix. Tolerances are unchanged.','',
            'The original orientation-only XREF24 NF4/BF16 strict geometry failures remain failed. Its future XREF22 stage is unrun under the prospectively published supervision amendment. Only its authenticated waiting supervisor is replaced; no model or numerical worker is stopped.','',
            'One serialized numerical CPU thread at Below Normal priority protects the math process. The XREF22 repair waits for the original XREF22 and both STORY23 lanes to finish. No causal/sham floor or power is calibrated, zero formal comparisons, no transfer/equivalence claim. The168-case test remains sealed and the continuous goal remains active.','',
            f"Source seal: `{payload['source_manifest_sha256']}`.",'',f'[Complete aggregate]({stem}.json) · [Repair protocol](../../experiments/nan_xref25/PROTOCOL.md) · [Original report](NAN_'+('XREF19' if source=='xref19' else 'XREF22')+'_PROGRESS.md)','']
        (dest/('NAN_XREF25_'+source.upper()+'_PROGRESS.md')).write_text('\n'.join(lines),encoding='utf-8',newline='\n')


if __name__=='__main__':main()
