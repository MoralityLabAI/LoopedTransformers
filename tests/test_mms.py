import copy
import torch
from research_gym.interp.mms import GradientBank, MechanisticMemory


def test_cache_budget_collision_age_and_resume():
    b = GradientBank(0.00004, (4,))
    assert b.capacity == 1
    b.write(torch.tensor([0, 1]), torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]]), 0)
    v, hit = b.read(torch.tensor([0, 1]), 1, 2)
    assert hit.tolist() == [False, True]
    assert v[1].tolist() == [5., 6., 7., 8.]
    assert not b.read(torch.tensor([1]), 0, 2)[1].any()
    assert not b.read(torch.tensor([1]), 3, 2)[1].any()
    assert b.allocated_bytes <= b.budget_bytes
    c = GradientBank(0.00004, (4,))
    c.load_state_dict(copy.deepcopy(b.state_dict()))
    assert torch.equal(c.read(torch.tensor([1]), 2, 2)[0], b.read(torch.tensor([1]), 2, 2)[0])


def test_observe_parity_and_feedback():
    torch.manual_seed(5)
    torch.set_num_threads(1)
    layer = torch.nn.Linear(8, 8)
    base = copy.deepcopy(layer)
    feedback = copy.deepcopy(layer)
    x, keys = torch.randn(5, 8), torch.arange(5)
    a = MechanisticMemory(layer, mib=.02, mode='observe', rank=4, topk=2)
    b = MechanisticMemory(feedback, mib=.02, mode='bpd', rank=4, topk=2)
    for step in range(2):
        for model in (base, layer, feedback):
            model.zero_grad()
        base(base(x)).square().mean().backward()
        with a.capture(keys, step, 0, 2):
            layer(layer(x)).square().mean().backward()
        with b.capture(keys, step, 0, 2):
            feedback(feedback(x)).square().mean().backward()
        assert torch.equal(layer.weight.grad, base.weight.grad)
        assert torch.equal(feedback.weight.grad, base.weight.grad) == (step == 0)
    assert b.admitted > 0
    assert b.bank.hits == 5
    assert not b.bank.pages[0][0].requires_grad


def test_resume_next_gradient():
    torch.manual_seed(19)
    x, keys = torch.randn(3, 8), torch.arange(3)
    a = torch.nn.Linear(8, 8)
    m = MechanisticMemory(a, mib=.02, rank=4)
    with m.capture(keys, 0, 0, 2):
        a(a(x)).square().mean().backward()
    b = copy.deepcopy(a)
    n = MechanisticMemory(b, mib=.02, rank=4)
    n.load_state_dict(copy.deepcopy(m.state_dict()))
    a.zero_grad()
    b.zero_grad()
    with m.capture(keys, 1, 0, 2):
        a(a(x)).square().mean().backward()
    with n.capture(keys, 1, 0, 2):
        b(b(x)).square().mean().backward()
    assert torch.equal(a.weight.grad, b.weight.grad)


def test_rmp_actual_decoder_observe_and_resume_identity():
    from research_gym.interp.training import CellConfig, cell_identity, cell_loss
    from research_gym.interp.models import build_model
    torch.manual_seed(31)
    config = CellConfig(arm='LOOP-T', seed=31, families=('pointer_chase',),
                        hidden_size=16, num_heads=4, visits=3, batch_per_family=4,
                        deep_supervision=True)
    assert not any(k.startswith('mms_') for k in cell_identity(config))
    model = build_model(config.arm, hidden_size=config.hidden_size, num_heads=config.num_heads,
                        visits=config.visits, beta=config.beta)
    baseline = copy.deepcopy(model)
    memory = MechanisticMemory(model.blocks[0].mlp[-1], mib=.1, mode='observe', rank=4)
    tokens, targets = torch.randint(1,32,(4,8)), torch.randint(1,32,(4,))
    keys = torch.arange(4)
    for step in range(2):
        model.zero_grad()
        baseline.zero_grad()
        cell_loss(baseline,tokens,targets,config).mean().backward()
        with memory.capture(keys,step,1,3):
            cell_loss(model,tokens,targets,config).mean().backward()
        for p,q in zip(model.parameters(),baseline.parameters(),strict=True):
            assert torch.equal(p.grad,q.grad)
    assert memory.bank.hits == 4


def test_small_rank_proxy_can_admit_below_point_five():
    torch.manual_seed(44)
    layer = torch.nn.Linear(64,32)
    m = MechanisticMemory(layer,mib=.02,rank=1,topk=1)
    x = torch.randn(5,64)
    keys = torch.arange(5)
    for step in range(2):
        layer.zero_grad()
        with m.capture(keys,step,0,1):
            layer(x).square().mean().backward()
    assert m.map_cosine < .5
    assert m.admitted > 0


def test_streamed_checkpoint_hash_matches_original(tmp_path):
    import hashlib
    from research_gym.architecture_discovery.checkpoint import _sha256_file
    path = tmp_path/'checkpoint'
    path.write_bytes(b'abcd'*1500000)
    assert _sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_shared_bank_lru_and_multisite_resume():
    from research_gym.interp.mms import MultiSiteMemory
    torch.manual_seed(93)
    a,b = torch.nn.Linear(8,8),torch.nn.Linear(8,8)
    # 60 rows at 32 bytes each; two active pages from distinct sites must compete.
    memory = MultiSiteMemory({'a':a,'b':b},mib=.0019,mode='observe',rank=4)
    keys=torch.arange(32)
    x=torch.randn(32,8)
    for step in range(4):
        a.zero_grad();b.zero_grad()
        with memory.capture(keys,step,0,1):
            b(a(x)).square().mean().backward()
        assert memory.pool.used <= memory.pool.budget_bytes
        assert memory.pool.used == sum(m.bank.allocated_bytes for m in memory.sites.values() if m.bank)
    assert memory.pool.evictions>0
    assert memory.selections=={'a':2,'b':2}
    c,d=copy.deepcopy(a),copy.deepcopy(b)
    restored=MultiSiteMemory({'a':c,'b':d},mib=.0019,mode='observe',rank=4)
    restored.load_state_dict(copy.deepcopy(memory.state_dict()))
    assert restored.pool.state_dict()==memory.pool.state_dict()
    a.zero_grad();b.zero_grad();c.zero_grad();d.zero_grad()
    with memory.capture(keys,4,0,1):
        b(a(x)).square().mean().backward()
    with restored.capture(keys,4,0,1):
        d(c(x)).square().mean().backward()
    assert torch.equal(a.weight.grad,c.weight.grad)
    assert memory.pool.state_dict()==restored.pool.state_dict()


def test_adaptive_schedule_explores_all_sites():
    from research_gym.interp.mms import MultiSiteMemory
    m=MultiSiteMemory({str(i):torch.nn.Linear(4,4) for i in range(4)},mib=.1,schedule='adaptive')
    chosen=[m.select(s) for s in range(16)]
    assert [chosen[i] for i in (0,4,8,12)] == ['0','1','2','3']
    prior=dict(m.selections)
    assert m.select(15)==chosen[15]
    assert prior==m.selections


def test_hash_pinned_multisite_map_manifest(tmp_path):
    import json
    import hashlib
    from research_gym.interp.mms import read_map_manifest, MultiSiteMemory
    layer=torch.nn.Linear(8,4)
    v=torch.eye(8)[:,:4];u=torch.eye(4)
    source=tmp_path/'factors.pt'
    torch.save({'V':v,'U':u},source)
    manifest=tmp_path/'maps.json'
    manifest.write_text(json.dumps({'mlp':{'path':'factors.pt',
                          'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}}))
    sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
    maps=read_map_manifest(manifest,sha)
    memory=MultiSiteMemory({'mlp':layer},mib=.1,maps=maps)
    assert memory.sites['mlp'].map_kind=='native_export_unverified_semantics'
    source.write_bytes(b'changed')
    import pytest
    with pytest.raises(ValueError): read_map_manifest(manifest,sha)


def test_canonical_factor_rescaling_and_permutation_invariance(tmp_path):
    """Equivalent factorizations (rescaled, sign-flipped, permuted) must inject identical gradients."""
    import hashlib
    from research_gym.interp.mms import MechanisticMemory
    torch.manual_seed(2026)
    layer_a = torch.nn.Linear(12, 16)
    layer_b = copy.deepcopy(layer_a)
    v0 = torch.randn(12, 6)
    u0 = torch.randn(6, 16)
    # Match layer weight so reconstruction cosine is 1.0
    with torch.no_grad():
        layer_a.weight.copy_((v0 @ u0).T)
        layer_b.weight.copy_((v0 @ u0).T)
    scales = torch.tensor([1e-3, -40.0, 0.25, 250.0, -0.02, 7.0])
    perm = torch.tensor([4, 1, 5, 0, 3, 2])
    v1 = (v0 * scales[None, :])[:, perm]
    u1 = (u0 / scales[:, None])[perm, :]
    fa, fb = tmp_path / 'fa.pt', tmp_path / 'fb.pt'
    torch.save({'V': v0, 'U': u0}, fa)
    torch.save({'V': v1, 'U': u1}, fb)
    ma = MechanisticMemory(layer_a, mib=0.5, mode='bpd', rank=6, topk=2, ratio=0.2,
                           map_file=str(fa), map_sha=hashlib.sha256(fa.read_bytes()).hexdigest())
    mb = MechanisticMemory(layer_b, mib=0.5, mode='bpd', rank=6, topk=2, ratio=0.2,
                           map_file=str(fb), map_sha=hashlib.sha256(fb.read_bytes()).hexdigest())
    keys = torch.arange(8)
    x = torch.randn(8, 12)
    for step in (0, 1):
        layer_a.zero_grad(); layer_b.zero_grad()
        with ma.capture(keys, step, 0, 1):
            layer_a(x).square().mean().backward()
        with mb.capture(keys, step, 0, 1):
            layer_b(x).square().mean().backward()
    assert torch.allclose(layer_a.weight.grad, layer_b.weight.grad, atol=1e-5)
    assert ma.admitted == mb.admitted and ma.admitted > 0
