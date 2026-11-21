"""Sixteen actual optimizer steps per depth on a tiny random CPU model only."""
import argparse
from dataclasses import asdict
import math
from .common import ROOT, read, sha, write


def main():
    p = argparse.ArgumentParser(); p.add_argument('--bpd-root', required=True); p.add_argument('--out', required=True); a = p.parse_args()
    from experiments.nan_ranksite13.resources import below_normal
    below_normal()
    import torch
    from mechinterp.nanbeige_bpd_gym.native_components import load_bpd
    from experiments.nan_fit07.worker import make_config
    torch.set_num_threads(1); framework = load_bpd(a.bpd_root); cfg = read(ROOT / 'config.json')
    cfg.update(site='proj', maximum_prompt_tokens=7); results = []
    for passes in (1, 2):
        torch.manual_seed(cfg['source_seed'])
        model = torch.nn.Module(); model.proj = torch.nn.Linear(32, 32, bias=False); model.head = torch.nn.Linear(32, 48, bias=False)
        contract = make_config(framework['config'], cfg, passes)
        bpd = framework['components'].BPDModel(model, {'proj': 8}, max_passes=passes, ablation=contract.ablation, init_svd=True)
        def adapter(batch, n_passes, pass_offset, init_state):
            h = batch if init_state is None else init_state; logits = {}; boundaries = {}
            for t in range(pass_offset, pass_offset + n_passes):
                h = torch.tanh(h + model.proj(h)); logits[t] = model.head(h[:, -1:]); boundaries[t] = h[:, -1:]
            return logits, boundaries, None, None
        trainer = framework['trainer'].BPDTrainer(bpd, contract, adapter, is_causal_ci=True, device='cpu')
        batches = [torch.randn(1, 7, 32) for _ in range(16)]
        for batch in batches:
            trainer.run_clean_rollout(batch, update_means=True)
        trainer.pass_tolerances = {t: 1. for t in range(passes)}
        before = [p.detach().clone() for p in bpd.component_parameters()]; ci_before = [p.detach().clone() for p in trainer.ci_net.parameters()]
        metrics = [asdict(trainer.train_step(batch, i)) for i, batch in enumerate(batches)]
        for r in metrics:
            assert all(math.isfinite(v) for v in r.values() if isinstance(v, (int, float)))
            total = cfg['coeff_faith'] * r['faith_loss'] + r['min_loss'] + cfg['coeff_stoch'] * r['stoch_loss'] + cfg['coeff_adv'] * r['adv_loss']
            assert math.isclose(total, r['total_loss'], rel_tol=2e-6, abs_tol=1e-6)
        factor_update = sum(float((p.detach()-v).square().sum()) for p, v in zip(bpd.component_parameters(), before))**.5
        ci_update = sum(float((p.detach()-v).square().sum()) for p, v in zip(trainer.ci_net.parameters(), ci_before))**.5
        assert factor_update > 0 and ci_update > 0
        for parameter in bpd.component_parameters() + list(trainer.ci_net.parameters()):
            assert trainer.optimizer.state[parameter]['step'].item() == 16
        results.append(dict(passes=passes, optimizer_steps=16, factor_update_L2=factor_update, controller_update_L2=ci_update, finite_losses=True, four_term_accounting=True))
    receipt = dict(status='passed', results=results, pretrained_model_loaded=False, CUDA_used=False,
        scope='Tiny random dense CPU stock-trainer scheduling/loss/update fixture; not native NF4 or pretrained fit validation. Pass tolerances fixed to1.',
        source_sha256=sha(ROOT / 'validate_stock_cpu.py'), sealed_test_accessed=False)
    write(a.out, receipt); print(receipt, flush=True)


if __name__ == '__main__':
    main()
