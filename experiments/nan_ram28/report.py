"""Full inherited coverage tables with explicit new resource-condition provenance."""
import argparse
import json
from pathlib import Path
from .common import ROOT,RUNS,read,sha

OUTPUTS=('reports/nanbeige_bpd_gym/NAN_RAM28_PROGRESS.md','reports/nanbeige_bpd_gym/nan_ram28_progress.json')

def run(artifact):
    import experiments.nan_mlp27.report as method
    method.ROOT=ROOT;method.RUNS=RUNS
    staging=RUNS/'report_staging';method.run(artifact,staging)
    data=read(staging/'nan_mlp27_progress.json')
    data.update(experiment='NAN-RAM-28',source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),
        method_source_manifest_sha256=sha('experiments/nan_mlp27/SOURCE_MANIFEST.json'),
        qualification=dict(status='explicit_user_host_RAM_resource_condition',run_both_backends=False),
        BF16_status='deferred_host_RAM_policy',CPU_audits_overlap_GPU=False)
    data['terminal_coverage']['bf16']['status']='deferred_host_RAM_policy'
    attestation=RUNS/'loader_attestation.json'
    if attestation.exists():data['loader_attestation']=read(attestation)
    source=(staging/'NAN_MLP27_PROGRESS.md').read_text(encoding='utf-8')
    tables=source.partition('All registered task/control strata are shown.')[2]
    assert tables
    lines=['# NAN-RAM-28: GPU-resident NF4 full gym with bounded host RAM','',
        'New source-frozen resource condition after the user requested less host-memory pressure. It uses file-backed BF16 checkpoint tensors, quantizes decoder layers with the original NF4 conversion, and retains the GPU head. It does not run the old conditional MLP27 registration or claim its qualification.', '',
        'Scope remains all 1,549 baseline trajectories and 216 true batched clean/intervention A/B pairs for NF4. MLP gate/up workspaces use 256-token chunks and one physical full-token down projection. Existing padded capacity/closure checks precede scored episodes. All 40 states, native reached outputs and masks are retained at every generated prefix. Fixed SAE/random controls and independent BPD audits use the original numerical methods after GPU termination.', '',
        'BF16 is deferred under the host-RAM policy; its missing results remain unknown. The interrupted CTX21 BF16 run, original failed terminal audit and all closed files remain preserved. No original receipt was synthesized and no reply was replayed.', '',
        'Host limits: start with at least 4 GiB available; check for at least 2 GiB available and at most 6 GiB private worker commitment between captures. A separate supervisor terminates only this authenticated worker if available RAM falls below 1 GiB or private commitment exceeds 6.5 GiB. The later host audit has a 1.5 GiB private budget and 2 GiB system reserve. One CPU thread, Below Normal priority; no concurrent substantial host audit. The math process is untouched.', '',
        'These resource thresholds are operating budgets, not scientific competence gates. The original 168-case test stays sealed. Repeated sources/tokens/routes/features are not independent samples; no transfer, equivalence or causal confirmation is claimed. The continuous goal remains active.', '']
    for backend in ('nf4','bf16'):
        c=data['terminal_coverage'][backend]
        lines.append(f"{backend.upper()}: {'deferred_host_RAM_policy' if backend=='bf16' else c['status']}; closed baseline {c.get('baseline_completed','unknown')}/1549; A/B {c.get('AB_pairs_completed','unknown')}/216.")
    lines+=['','All registered task/control strata are shown.'+tables,'',
        '[Resource protocol](../../experiments/nan_ram28/PROTOCOL.md) · [Complete aggregate](nan_ram28_progress.json)','']
    Path(OUTPUTS[0]).write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    Path(OUTPUTS[1]).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',required=True);run(p.parse_args().artifact_root)
