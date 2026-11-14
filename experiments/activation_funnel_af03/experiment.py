"""AF-03 frozen-codec-width native TypedLoop causal-interchange training and scoring.

Training touches only AF-02 canonical train cases, using development inputs for
eligibility checks. Evaluation runs separately on held-out cases after saving all
student weights. Reports retain every seed/arm, including failed scientific gates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.activation_funnel_af02 import gym
from experiments.activation_funnel_af02 import train as af02train
from experiments.activation_funnel_af02.model import NativeTeacher
from .core import TokenCodec, paired, audited_cross_entropy, donor_roster, synthetic_extra_waits

ARMS = ('task', 'status', 'causal')
FAMILIES = ('depth_context', 'delay2', 'delay4', 'delay5', 'delay8')
SEEDS = (17, 29, 43)
WIDTHS = (1, 2)
ROOT = Path(__file__).resolve().parent
DEFAULT_STEPS = 240
BATCH_SIZE = 32
LR = 0.002
WEIGHTS = dict(status=0.5, paired_task=0.5, paired_status=0.5,
               locality=0.25, consistency=0.1)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def save_jsonl(path, rows):
    with Path(path).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def model_and_codec(source_state, bases, width):
    model = NativeTeacher()
    model.load_state_dict(source_state, strict=True)
    codec = TokenCodec(width)
    for token in ('atom', 'rule'):
        item = bases[token]
        getattr(codec, token).initialize_from_basis(item['mean'], item['basis'][:, :width])
    return model, codec


def train_only_bases(source, cases):
    traces = [None] * len(cases)
    source.eval()
    with torch.no_grad():
        for indices in af02train.bucket_indices(cases).values():
            x, _, _ = af02train.tensor_data([cases[i] for i in indices])
            h = source.rollout(x).states
            for j, idx in enumerate(indices):
                traces[idx] = h[j].detach().clone()
    return af02train.fit_bases('native', traces)


def grouped_indices(cases, indices):
    groups = defaultdict(list)
    for idx in indices:
        c = cases[idx]
        groups[(len(c.schedule), gym.first_probe_cut(c))].append(idx)
    return list(groups.values())


def sample_pairs(roster, indices, generator):
    pick = torch.randint(2, (len(indices),), generator=generator).tolist()
    return [roster[i][j] for i, j in zip(indices, pick)]


def causal_losses(model, codec, receivers, donors, pairs, rx, audit):
    first = torch.tensor([p.first_probed_branch for p in pairs], dtype=torch.long)
    cuts = {p.cut_event for p in pairs}
    if len(cuts) != 1:
        raise ValueError('all visits in a training bucket must share the cut')
    result = paired(model, codec, receivers, donors, first, cut_event=next(iter(cuts)),
                    condition='addressed', direct=True)
    target_task = torch.tensor([p.expected_action for p in pairs], dtype=torch.long)
    target_audit = torch.tensor([p.audit_target for p in pairs], dtype=torch.long)
    other = 1 - first
    row = torch.arange(len(first))
    paired_task = F.cross_entropy(result.task_logits, target_task)
    paired_status = audited_cross_entropy(result.audit_logits, target_audit)
    locality = F.cross_entropy(result.audit_logits[row, other], audit[row, other])
    # Stop-gradient direct hybrid serves as a same-model causal-commutation target;
    # oracle labels keep this auxiliary target from defining correctness.
    consistency = F.mse_loss(F.softmax(result.audit_logits, -1),
                             F.softmax(result.direct_audit_logits.detach(), -1))
    consistency += F.mse_loss(F.softmax(result.task_logits, -1),
                              F.softmax(result.direct_task_logits.detach(), -1))
    return {'paired_task': paired_task, 'paired_status': paired_status,
            'locality': locality, 'consistency': consistency}


def train_one(af02_folder, output, seed, width, steps=DEFAULT_STEPS, smoke=False):
    if seed not in SEEDS or width not in WIDTHS or (not smoke and steps != DEFAULT_STEPS):
        raise ValueError('attempted training configuration differs from registration')
    if not smoke and torch.__version__ != '2.5.1+cpu':
        raise RuntimeError('primary AF-03 requires registered Torch 2.5.1+cpu')
    torch.set_num_threads(2)
    output = Path(output)
    if output.exists():
        raise FileExistsError('AF-03 training requires fresh output')
    output.mkdir(parents=True)
    manifest = af02train.load_frozen(Path(af02_folder))
    source, _ = af02train.load_model_arm(Path(af02_folder), 'native', seed, 'teacher', manifest)
    source_state = copy.deepcopy(source.state_dict())
    source_file = af02train.checkpoint_name('native', 'teacher', seed)
    source_sha = manifest['files'][source_file]['sha256']
    train_cases, dev_cases = gym.canonical_cases('train'), gym.canonical_cases('dev')
    roster = donor_roster(train_cases)
    bases = train_only_bases(source, train_cases)
    torch.manual_seed(20000 + seed)
    _, common_codec = model_and_codec(source_state, bases, width)
    common_codec_state = copy.deepcopy(common_codec.state_dict())
    training_results, predicted_rows, weights = {}, [], {}
    for arm in ARMS:
        torch.manual_seed(30000 + seed)
        model = NativeTeacher()
        model.load_state_dict(source_state, strict=True)
        codec = TokenCodec(width)
        codec.load_state_dict(common_codec_state, strict=True)
        model.train(); codec.train()
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(codec.parameters()),
                                      lr=LR, weight_decay=0.01)
        rng = torch.Generator().manual_seed(9000 + seed)
        logs, first_cell_grad, start = [], None, time.perf_counter()
        for step in range(steps):
            selected = torch.randint(len(train_cases), (BATCH_SIZE,), generator=rng).tolist()
            donors_for_step = sample_pairs(roster, selected, rng)
            # Each sampled occurrence, including duplicates, has an independent donor.
            entries = list(zip(selected, donors_for_step))
            buckets = defaultdict(list)
            for idx, p in entries:
                c = train_cases[idx]
                buckets[(len(c.schedule), gym.first_probe_cut(c))].append((idx, p))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros(())
            loss_parts = defaultdict(float)
            for group in buckets.values():
                cases = [train_cases[i] for i, _ in group]
                batch_pairs = [p for _, p in group]
                rx, task, audits = af02train.tensor_data(cases)
                dx, _, _ = af02train.tensor_data([train_cases[p.donor_index] for p in batch_pairs])
                trial = model.rollout(rx, codec)
                pieces = {'task': F.cross_entropy(trial.task_logits, task)}
                if arm != 'task':
                    pieces['status'] = audited_cross_entropy(trial.audit_logits, audits)
                if arm == 'causal':
                    pieces.update(causal_losses(model, codec, rx, dx, batch_pairs, trial, audits))
                bucket_loss = pieces['task']
                for name, weight in WEIGHTS.items():
                    if name in pieces:
                        bucket_loss = bucket_loss + weight * pieces[name]
                frac = len(group) / len(selected)
                loss = loss + frac * bucket_loss
                for key, value in pieces.items():
                    loss_parts[key] += frac * float(value.detach())
            loss.backward()
            if arm == 'causal' and first_cell_grad is None:
                first_cell_grad = sum(float(p.grad.detach().square().sum()) for p in model.cell.parameters()
                                      if p.grad is not None) ** .5
                if first_cell_grad <= 0:
                    raise RuntimeError('causal gradients did not reach the native TypedLoop cell')
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(codec.parameters()), 1.)
            optimizer.step()
            if step % 40 == 0 or step == steps - 1:
                logs.append({'step': step, 'loss': float(loss.detach()), **loss_parts})
        model.eval(); codec.eval()
        records, metrics = {}, {}
        for split, cases in (('train', train_cases), ('dev', dev_cases)):
            rows, summary = clean_score(model, codec, cases, split)
            predicted_rows.extend([{**row, 'seed': seed, 'width': width, 'arm': arm} for row in rows])
            metrics[split] = summary
        path = output / f'{arm}.pt'
        torch.save({'source_sha256': source_sha, 'seed': seed, 'width': width, 'arm': arm,
                    'model': model.state_dict(), 'codec': codec.state_dict()}, path)
        arrays = {'model.' + key: x.detach().numpy() for key, x in model.state_dict().items()}
        arrays.update({'codec.' + key: x.detach().numpy() for key, x in codec.state_dict().items()})
        np.savez(output / f'{arm}.npz', **arrays)
        weights[arm] = {name: {'sha256': file_hash(output / f'{arm}.{ext}'),
                               'bytes': (output / f'{arm}.{ext}').stat().st_size}
                        for ext, name in (('pt', 'torch'), ('npz', 'named_arrays'))}
        training_results[arm] = {'loss_history': logs, 'seconds': time.perf_counter() - start,
                                 'native_cell_first_step_grad_l2': first_cell_grad,
                                 'train': metrics['train'], 'dev': metrics['dev'],
                                 'total_parameters': sum(p.numel() for p in model.parameters()) +
                                                     sum(p.numel() for p in codec.parameters())}
        print(json.dumps({'stage': 'train', 'seed': seed, 'width': width, 'arm': arm,
                          'dev': metrics['dev'], 'seconds': training_results[arm]['seconds']}),flush=True)
    save_jsonl(output / 'train_dev_predictions.jsonl', predicted_rows)
    files = {'core.py': ROOT / 'core.py', 'experiment.py': ROOT / 'experiment.py',
             'PROTOCOL.json': ROOT / 'PROTOCOL.json',
             'AF02.PROTOCOL.json': ROOT.parent / 'activation_funnel_af02' / 'PROTOCOL.json',
             'AF02.corpus_manifest.json': ROOT.parent / 'activation_funnel_af02' / 'corpus_manifest.json'}
    receipt = {'experiment':'AF-03','scope':'smoke' if smoke else 'primary',
               'af02_source_checkpoint':source_file, 'af02_source_sha256':source_sha,
               'parent_training_manifest_sha256':file_hash(Path(af02_folder) / 'train_manifest.json'),
               'seed':seed,'token_code_width':width,'carry_coordinates':5*width,
               'source_coordinates':80, 'steps':steps, 'batch_size':BATCH_SIZE,
               'learning_rate':LR, 'torch_version':torch.__version__, 'threads':2,
               'fitting_splits':['train','dev_for_eligibility_only'],
               'objective_weights':WEIGHTS, 'source_file_sha256':{n:file_hash(p) for n,p in files.items()},
               'weights':weights, 'training':training_results,
               'train_dev_predictions': {'sha256':file_hash(output / 'train_dev_predictions.jsonl'),
                                         'rows':len(predicted_rows)}}
    save_json(output / 'train_receipt.json', receipt)
    (output / 'train_receipt.sha256').write_text(file_hash(output / 'train_receipt.json')+'\n')
    return receipt


def clean_score(model, codec, cases, family, extra=0):
    rows = [None] * len(cases)
    with torch.no_grad():
        groups = defaultdict(list)
        for i, case in enumerate(cases):
            x = synthetic_extra_waits(case, extra) if extra else torch.tensor(gym.feature_sequence(case))
            groups[len(x)].append((i, x))
        for items in groups.values():
            x = torch.stack([t for _, t in items]).float()
            trace = model.rollout(x, codec)
            for j, (i, _) in enumerate(items):
                case = cases[i]
                p = trace.task_logits[j].argmax().item()
                audit = trace.audit_logits[j].argmax(-1).tolist()
                rows[i] = {'case_id':case.case_id,'canonical_id':case.canonical_id,
                           'public_input_sha256':gym.public_input_hash(case),
                           'order': list(case.probe_order), 'family':family,
                           'visits':2 * x.shape[1], 'target':gym.target_action(case),
                           'audit_target':list(gym.audit_targets(case)),
                           'task_prediction':p, 'audit_prediction':audit,
                           'task_logits':trace.task_logits[j].tolist(),
                           'audit_logits':trace.audit_logits[j].tolist(),
                           'tainted_attempt':int(p < 2 and case.statuses[p] == 'tainted')}
    def summarize(part):
        return {'cases':len(part), 'task':sum(x['task_prediction']==x['target'] for x in part)/len(part),
                'audit_exact':sum(x['audit_prediction']==x['audit_target'] for x in part)/len(part),
                'audit_by_branch':[sum(x['audit_prediction'][j]==x['audit_target'][j] for x in part)/len(part) for j in (0,1)],
                'tainted_attempts':sum(x['tainted_attempt'] for x in part)}
    return rows, {**summarize(rows), 'by_order':{
        ','.join(map(str, order)):summarize([x for x in rows if tuple(x['order'])==order])
        for order in gym.ORDERS}}


def score_patches(model, codec, cases, family, extra=0):
    roster = donor_roster(cases)
    pairs = [p for arr in roster.values() for p in arr]
    groups = defaultdict(list)
    for p in pairs:
        groups[(p.cut_event, len(cases[p.receiver_index].schedule)+extra)].append(p)
    rows = []
    max_whole_diff = 0.
    with torch.no_grad():
        for (cut, length), group in groups.items():
            def x_for(i):
                c = cases[i]
                return synthetic_extra_waits(c, extra) if extra else torch.tensor(gym.feature_sequence(c),dtype=torch.float32)
            rx = torch.stack([x_for(p.receiver_index) for p in group])
            dx = torch.stack([x_for(p.donor_index) for p in group])
            first = torch.tensor([p.first_probed_branch for p in group], dtype=torch.long)
            results = {name:paired(model, codec, rx, dx, first, cut_event=cut,
                                   condition=name, direct=name=='whole')
                       for name in ('addressed','other','whole')}
            max_whole_diff = max(max_whole_diff, float((results['whole'].task_logits -
                             results['whole'].direct_task_logits).abs().max()),
                             float((results['whole'].audit_logits -
                             results['whole'].direct_audit_logits).abs().max()))
            for i,p in enumerate(group):
                row = {'pair_id':p.pair_id, 'family':family, 'first_branch':p.first_probed_branch,
                       'recipient_id':p.receiver_case_id,'donor_id':p.donor_case_id,
                       'hybrid_id':p.hybrid_case_id, 'changed_prefix_status':p.changed_prefix_status,
                       'target_task':p.expected_action,'target_audit':list(p.audit_target),
                       'recipient_audit':list(gym.audit_targets(cases[p.receiver_index])),
                       'cut_event':p.cut_event,'visits':length*2, 'conditions':{}}
                for name,res in results.items():
                    pred=res.task_logits[i].argmax().item()
                    ap=res.audit_logits[i].argmax(-1).tolist()
                    row['conditions'][name]={'task_prediction':pred,'audit_prediction':ap,
                                             'task_logits':res.task_logits[i].tolist(),
                                             'audit_logits':res.audit_logits[i].tolist(),
                                             'patched_code':res.patched_code[i].tolist()}
                rows.append(row)
    def accuracy(cond, kind):
        def correct(r):
            ap = r['conditions'][cond]['audit_prediction']
            f = r['first_branch'];other=1-f
            if kind=='first':return ap[f]==r['target_audit'][f]
            if kind=='other':return ap[other]==r['recipient_audit'][other]
            if kind=='exact':return ap==r['target_audit']
            return r['conditions'][cond]['task_prediction']==r['target_task']
        return sum(correct(r) for r in rows)/len(rows)
    summary={'changed_pairs':len(rows), 'whole_state_direct_max_error':max_whole_diff,
             'addressed':{key:accuracy('addressed',key) for key in ('first','other','exact','task')},
             'other_control':{key:accuracy('other',key) for key in ('first','other','exact','task')},
             'whole':{key:accuracy('whole',key) for key in ('first','other','exact','task')}}
    summary['selectivity'] = summary['addressed']['first']-summary['other_control']['first']
    return rows,summary


def qualified(metrics):
    return all(m['task'] >= 0.95 and all(a >= 0.95 for a in m['audit_by_branch'])
               for split in ('train','dev') for m in metrics[split]['by_order'].values())


def evaluate_one(training, output, seed, width, smoke=False):
    training=Path(training);output=Path(output)
    if output.exists():raise FileExistsError('AF-03 scoring output must be fresh')
    output.mkdir(parents=True)
    receipt=json.loads((training / 'train_receipt.json').read_text())
    if file_hash(training / 'train_receipt.json') != (training / 'train_receipt.sha256').read_text().strip():
        raise ValueError('training receipt checksum mismatch')
    if (receipt['seed'],receipt['token_code_width']) != (seed,width):
        raise ValueError('training identity mismatch')
    if not smoke and (receipt['scope']!='primary' or torch.__version__ != '2.5.1+cpu'):
        raise ValueError('primary evaluation needs frozen primary Torch2.5.1 receipt')
    for name,p in (('core.py',ROOT/'core.py'),('experiment.py',ROOT/'experiment.py'),
                   ('PROTOCOL.json',ROOT/'PROTOCOL.json'),
                   ('AF02.PROTOCOL.json',ROOT.parent/'activation_funnel_af02'/'PROTOCOL.json'),
                   ('AF02.corpus_manifest.json',ROOT.parent/'activation_funnel_af02'/'corpus_manifest.json')):
        if file_hash(p)!=receipt['source_file_sha256'][name]:
            raise ValueError('source changed since fitting: '+name)
    torch.set_num_threads(2)
    evaluated={}; predictions=[]; interventions=[]
    for arm in ARMS:
        for ext, name in (('pt','torch'),('npz','named_arrays')):
            path=training/f'{arm}.{ext}'
            if file_hash(path)!=receipt['weights'][arm][name]['sha256']:
                raise ValueError('checkpoint tampering: '+str(path))
        pkg=torch.load(training/f'{arm}.pt',map_location='cpu',weights_only=True)
        if (pkg['seed'],pkg['width'],pkg['arm'],pkg['source_sha256']) != (
                seed,width,arm,receipt['af02_source_sha256']):
            raise ValueError('checkpoint identity mismatch')
        model=NativeTeacher();model.load_state_dict(pkg['model'],strict=True)
        codec=TokenCodec(width);codec.load_state_dict(pkg['codec'],strict=True)
        with np.load(training/f'{arm}.npz',allow_pickle=False) as named:
            states={'model.'+k:v for k,v in model.state_dict().items()}
            states.update({'codec.'+k:v for k,v in codec.state_dict().items()})
            if set(named.files)!=set(states) or any(not np.array_equal(named[k],v.numpy()) for k,v in states.items()):
                raise ValueError('Torch/named-array tensor discrepancy')
        model.eval();codec.eval()
        arm_results={'training':receipt['training'][arm]}
        for family in FAMILIES:
            base=family if family in gym.SPLIT_SPEC else 'delay4'
            extra={'delay5':1,'delay8':4}.get(family,0)
            cases=gym.canonical_cases(base)
            clean,clean_metrics=clean_score(model,codec,cases,family,extra)
            patch,patch_metrics=score_patches(model,codec,cases,family,extra)
            predictions.extend([{**r,'seed':seed,'width':width,'arm':arm} for r in clean])
            interventions.extend([{**r,'seed':seed,'width':width,'arm':arm} for r in patch])
            if patch_metrics['whole_state_direct_max_error'] > 2e-5:
                raise RuntimeError('whole-code causal-control replay failed')
            arm_results[family]={'clean':clean_metrics,'patch':patch_metrics,
                                 'synthetic_timing_only':bool(extra)}
        arm_results['eligible']=qualified(arm_results['training']) if not smoke else False
        evaluated[arm]=arm_results
        print(json.dumps({'stage':'evaluation','seed':seed,'width':width,'arm':arm,
                          'eligible':arm_results['eligible'],
                          'depth':arm_results['depth_context']['patch'],
                          'delay8':arm_results['delay8']['clean']}),flush=True)
    # Registered seed-level gate: compare causal against status on two held-out families.
    status,causal=evaluated['status'],evaluated['causal']
    eligible=status['eligible'] and causal['eligible']
    checks={}
    for family in ('depth_context','delay4'):
        s,c=status[family],causal[family]
        checks[family]={
            'lift':c['patch']['addressed']['first']-s['patch']['addressed']['first'],
            'task_retention':c['clean']['task']-s['clean']['task'],
            'other_retention':c['patch']['addressed']['other'],
            'selectivity':c['patch']['selectivity'],
            'extra_tainted':c['clean']['tainted_attempts']-s['clean']['tainted_attempts']}
    pass_gate=all(x['lift']>=0.05-1e-10 and x['task_retention']>=-0.02-1e-10 and
                  x['other_retention']>=0.95 and x['selectivity']>=0.20 and x['extra_tainted']<=0
                  for x in checks.values())
    report={'experiment':'AF-03','seed':seed,'token_code_width':width,
            'carried_coordinates':5*width,'train_receipt_sha256':file_hash(training/'train_receipt.json'),
            'source_sha256':receipt['af02_source_sha256'],
            'primary_gate':('pass' if pass_gate else 'fail') if eligible else 'inconclusive',
            'gate_eligible':eligible, 'checks':checks, 'arms':evaluated,
            'synthetic_family_caveat':'delay5 and delay8 are tensor-only extra waits; no host proof credit'}
    save_jsonl(output/'clean_predictions.jsonl',predictions)
    save_jsonl(output/'counterfactuals.jsonl',interventions)
    report['file_hashes']={name:file_hash(output/name) for name in ('clean_predictions.jsonl','counterfactuals.jsonl')}
    save_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','evaluate'])
    parser.add_argument('--af02-checkpoints',type=Path)
    parser.add_argument('--train',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--width',type=int,required=True)
    parser.add_argument('--steps',type=int,default=DEFAULT_STEPS)
    parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args()
    if a.command=='train':
        if not a.af02_checkpoints:parser.error('train requires --af02-checkpoints')
        train_one(a.af02_checkpoints,a.out,a.seed,a.width,a.steps,a.smoke)
    else:
        if not a.train:parser.error('evaluate requires --train')
        evaluate_one(a.train,a.out,a.seed,a.width,a.smoke)


if __name__=='__main__':main()
