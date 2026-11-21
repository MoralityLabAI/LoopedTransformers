"""Validate preserved partial optimizer state without rescuing missing losses."""
from pathlib import Path
from .common import ROOT, RUNS, read, rows, sha, write


def main():
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    import numpy as np
    base = Path('runs/nan_fit16/nf4')
    assert any(r['event'] == 'worker_receipt_closed' for r in rows(base / 'events.jsonl'))
    audited = read('runs/nan_fit16/audits/nf4.json'); assert audited['status'] == 'passed'
    terminal = read(base / 'receipt.json'); assert audited['receipt_sha256'] == sha(base / 'receipt.json')
    results = []
    for passes in (1, 2):
        folder = base / f't{passes}_layer21_down_proj_rank8'; file = folder / 'receipt.json'
        if not file.exists():
            results.append(dict(passes=passes, status='unrun_or_no_terminal_cell')); continue
        r = read(file); item = dict(passes=passes, status=r['status'], optimizer_steps_completed=r['optimizer_steps_completed'], receipt_sha256=sha(file))
        if 'partial_checkpoint' in r:
            path = folder / r['partial_checkpoint']['file']; assert sha(path) == r['partial_checkpoint']['sha256']
            with np.load(path, allow_pickle=False) as z:
                assert all(np.isfinite(z[name]).all() for name in z.files)
                assert z['V'].shape == (10752, 8) and z['U'].shape == (8, 3072)
                for name in ('optimizer__factor_V__step', 'optimizer__factor_U__step', 'optimizer__ci_proj_in.weight__step', 'optimizer__ci_out_proj.weight__step'):
                    assert z[name].item() == r['optimizer_steps_completed']
            logged = read(folder / 'steps_local.json') if (folder / 'steps_local.json').exists() else []
            item.update(partial_checkpoint_sha256=sha(path), optimizer_counters_verified=True, logged_step_records=len(logged),
                missing_logged_losses=r['optimizer_steps_completed']-len(logged), failure_type=r.get('failure_type'),
                no_missing_loss_or_response_rescued=True)
        results.append(item)
    write(RUNS / 'original_attempt_audit.json', dict(status='passed', original_source_seal='35371bbb5323be88ae4eca4cb43677c2f583494b8a46a5f7b7aa89c2fc710139',
        receipt_sha256=sha(base / 'receipt.json'), NF4_cells=results, original_BF16='unrun_registered_supervision_amendment',
        XREF17='unrun_registered_supervision_amendment', neural_replay=False, sealed_test_accessed=False))
    print(dict(status='passed', cells=results), flush=True)


if __name__ == '__main__':
    main()
