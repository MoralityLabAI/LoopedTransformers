"""Publishable operational snapshots without reading partial neural answers."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime,timezone

def snapshot(repo,runs):
    experiment=repo/'experiments/nan_diag04';cfg=json.loads((experiment/'config.json').read_text());expected=cfg['calibration_roots']+cfg['development_roots'];backends={}
    for b in ('bf16','nf4'):
        path=runs/b;done=sorted(int(re.fullmatch(r'root_([0-9]+)\.json',p.name)[1]) for p in path.glob('root_*.json'))
        if not set(done)<=set(expected):raise ValueError('Unexpected root file')
        receipt=path/'receipt.json';r=json.loads(receipt.read_text()) if receipt.exists() else None
        phase=r['status'] if r else ('evaluating' if (path/'manifest.json').exists() else ('loading' if path.exists() else 'not_started'))
        backends[b]={'phase':phase,'completed_roots':done,'planned_roots':len(expected),'CAL_roots_completed':len(set(done)&set(cfg['calibration_roots'])),
            'DEV_roots_completed':len(set(done)&set(cfg['development_roots']))}
        if r:backends[b]|={k:r[k] for k in ('elapsed_seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes')}
    return {'schema_version':1,'updated_UTC':datetime.now(timezone.utc).isoformat(),'experiment':'NAN-DIAG-04','registration_commit':'10936e4caac06115e4a53b600cbab658295cc246',
        'source_manifest_sha256':hashlib.sha256((experiment/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),'backends':backends,
        'answer_data_opened_by_monitor':False,'scientific_readout':'Withheld until both backends finish and grading/audit complete',
        'BPD_transfer_attempted':False,'sealed_168_case_test_accessed':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--runs',default='runs/nan_diag04');p.add_argument('--out',required=True);p.add_argument('--report-url');a=p.parse_args()
    s=snapshot(Path(a.repo),Path(a.runs));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.report_url:
        audit=json.loads((Path(a.repo)/'experiments/nan_diag04/receipts/independent_audit.json').read_text())
        assert audit['status']=='passed' and all(v['phase']=='completed' for v in s['backends'].values())
        s['scientific_readout']='[Audited diagnostic report]('+a.report_url+')'
        s['audit_status']='passed'
    (out/'progress.json').write_text(json.dumps(s,indent=2)+'\n',encoding='utf-8',newline='\n')
    lines=['# Nanbeige BPD gym operational progress','',f"Updated {s['updated_UTC']}",'',
        'The registered NAN-DIAG-04 diagnostic separates reciprocal coefficient computation, raw operator selection, supplied-reference matching, neutral-code matching and copy controls.','',
        '| Backend | Phase | CAL complete | DEV complete |','|---|---|---:|---:|']
    for b,r in s['backends'].items():lines.append(f"| {b.upper()} | {r['phase']} | {r['CAL_roots_completed']}/2 | {r['DEV_roots_completed']}/8 |")
    lines+=['','[Pre-inference registration](https://github.com/MoralityLabAI/LoopedTransformers/commit/10936e4caac06115e4a53b600cbab658295cc246).',
        '',s['scientific_readout']+'. This monitor reads root filenames and execution receipts, not partial neural predictions.',
        '','No BPD transfer has run. The original 168-case ARG-03 test remains sealed.',
        '','Wider-suite work: Prime verifiers adapter and same-site physical SAE/VPD/BPD geometry bridge are separate integration work. No live Prime model evaluation, trained Nanbeige SAE result, or causal transfer is claimed here.','']
    (out/'STATUS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n');print(json.dumps(s['backends']))

if __name__=='__main__':main()
