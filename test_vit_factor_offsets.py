"""Token factor-offset contraction, initialization, gradients and schedule prefixes."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from condlr import BasisMixLowRank
from train_cifar10 import Config, build_model, main, make_optimizer, model_stats


def test_factor_biases_use_adapter_lr_without_weight_decay():
    cfg=Config(variant='dplr',factor_offsets=True,width=24,depth=1,heads=3,rank=2,n_basis=4,hyper_lr_multiplier=4.)
    model=build_model(cfg);optimizer=make_optimizer(model,cfg)
    groups={id(p):g for g in optimizer.param_groups for p in g['params']}
    assert len(groups)==len(list(model.parameters()))
    offsets=[p for name,p in model.named_parameters() if name.endswith(('U_offset','V_offset'))]
    assert len(offsets)==2 and all(groups[id(p)]['weight_decay']==0 and groups[id(p)]['lr']==cfg.lr*4 for p in offsets)
    assert groups[id(model.blocks[0].mlp.fc1.transform.lowrank.U)]['weight_decay']==cfg.weight_decay


def test_diagnostics_bypass_a_compiled_wrapper():
    from train_cifar10 import SyntheticImages
    from vit_quality_diagnostics import TokenDiagnostics
    cfg = Config(variant='dplr',factor_offsets=True,width=24,depth=1,heads=3,rank=2,n_basis=4,diagnostic_steps='0,1')
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self._orig_mod=model
        def forward(self, *args):
            raise AssertionError('Diagnostic probes must not call compiled forward')
    wrapped=Wrapper(build_model(cfg))
    diag=TokenDiagnostics(wrapped,cfg,SyntheticImages(8,10),torch.device('cpu'))
    assert diag.collect(wrapped,0)['step']==0


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('broadcast', [False, True])
@pytest.mark.parametrize('detach', [False, True])
def test_token_contractions_match_materialized_factors_and_all_gradients(broadcast, detach):
    m = BasisMixLowRank(5, 7, 5, rank=2, n_basis=3, factor_offsets=True).double()
    with torch.no_grad():
        m.V.normal_(); m.U_offset.normal_(); m.V_offset.normal_()
    x = torch.randn(2, 4, 5, dtype=torch.double, requires_grad=True)
    c = x.mean(1) if broadcast else x
    if detach: c = c.detach()
    a, b = m.to_coef(c).chunk(2, -1)
    u = torch.einsum('...k,ikr->...ir', a, m.U.view(5, 3, 2)) + m.U_offset
    v = torch.einsum('...k,kro->...ro', b, m.V.view(3, 2, 7)) + m.V_offset
    if broadcast: u, v = u[:, None], v[:, None]
    expected = (x.unsqueeze(-2) @ u @ v).squeeze(-2)
    actual = m.delta(x, c)
    torch.testing.assert_close(actual, expected)
    values = [x, m.U, m.V, m.to_coef.weight, m.to_coef.bias, m.U_offset, m.V_offset]
    ga = torch.autograd.grad(actual.square().sum(), values, retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), values)
    for a, b in zip(ga, gb): torch.testing.assert_close(a, b)


def test_zero_start_rng_parameter_increment_bf16_and_gradient_wakeup():
    torch.manual_seed(4); a = BasisMixLowRank(8, 16, 8, rank=2, n_basis=4); rng = torch.get_rng_state()
    torch.manual_seed(4); b = BasisMixLowRank(8, 16, 8, rank=2, n_basis=4, factor_offsets=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert sum(p.numel() for p in b.parameters()) - sum(p.numel() for p in a.parameters()) == 48
    x = torch.randn(2, 5, 8)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        delta = b.delta(x, x)
    assert delta.dtype == torch.bfloat16 and torch.count_nonzero(delta) == 0
    opt = torch.optim.SGD(b.parameters(), lr=.1)
    for _ in range(3):
        opt.zero_grad(); b(x, x).square().mean().backward(); opt.step()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in b.parameters())


def test_offsets_keep_rank_sized_intermediates_and_expected_flops():
    cfg = Config(variant='dplr', width=24, heads=3, depth=2, rank=2, n_basis=4)
    torch.manual_seed(3); a = build_model(cfg)
    torch.manual_seed(3); b = build_model(replace(cfg, factor_offsets=True))
    assert model_stats(b, cfg)['parameters'] - model_stats(a, cfg)['parameters'] == 2*2*(24+96)
    layer = b.blocks[0].mlp.fc1
    assert layer.additional_matmul_flops_per_token() - a.blocks[0].mlp.fc1.additional_matmul_flops_per_token() == 2*2*(24+96)
    x = torch.randn(2, 5, 24, requires_grad=True); saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(tuple(t.shape)) or t, lambda t: t):
        layer(x, x).sum().backward()
    assert (2,5,24,2) not in saved and (2,5,2,96) not in saved
    assert (2,5,4,2) in saved


def test_pilot_is_an_exact_prefix_with_diagnostics_and_future_lr_horizon(tmp_path):
    common = Config(variant='dplr', factor_offsets=True, width=24, heads=3, depth=1,
                    rank=2, n_basis=4, synthetic=True, synthetic_samples=12, val_limit=8,
                    epochs=4, batch_size=4, lr_schedule_steps=10, warmup_ratio=.2,
                    device='cpu', precision='fp32', num_workers=0, cpu_threads=2,
                    log_every=1, benchmark_warmup_steps=0)
    results = []
    for budget in (3, 10):
        cfg = replace(common, max_steps=budget, output_dir=str(tmp_path/str(budget)),
                      diagnostic_steps='0,1,3' if budget==3 else '0,1,3,10')
        path = tmp_path/f'{budget}.json'; path.write_text(json.dumps(asdict(cfg)))
        main(['--override_json', str(path)])
        results.append([json.loads(line) for line in (tmp_path/str(budget)/'metrics.jsonl').read_text().splitlines()])
    for key in ('lr','loss','accuracy','grad_norm_before_clip','hyper_grad_norm_before_clip'):
        assert [e[key] for e in results[0] if e['event']=='train'] == [e[key] for e in results[1] if e['event']=='train'][:3]
    a = next(e for e in results[0] if e['event']=='token_diagnostic' and e['step']==3)
    b = next(e for e in results[1] if e['event']=='token_diagnostic' and e['step']==3)
    assert a == b
    assert not list(tmp_path.rglob('*.pt'))
