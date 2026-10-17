"""Read-only raw-model probes for token-conditioned ViT training."""

from contextlib import contextmanager
import hashlib

import torch

from conditioning import ConditionedLinear, condition_input
from train_cifar10 import hyperparameter_ids


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def parameter_digest(model, adapter=False):
    ids = hyperparameter_ids(model)
    values = []
    for name, value in model.named_parameters():
        if (id(value) in ids) != adapter:
            continue
        if not adapter:
            for suffix in ('.transform.lowrank.base.', '.transform.base.', '.base.'):
                name = name.replace(suffix, '.')
        values.append((name, value))
    result = hashlib.sha256()
    for name, value in sorted(values):
        result.update(name.encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(tensor_digest(value).encode())
    return result.hexdigest()


def base_layer(layer):
    if not isinstance(layer, ConditionedLinear):
        return layer
    if layer.variant == 'dplr':
        return layer.transform.lowrank.base
    if layer.variant in ('basis', 'exact'):
        return layer.transform.base
    return layer.base


def lowrank_delta(layer, x, c):
    if isinstance(layer, ConditionedLinear):
        c = condition_input(c, layer.condition_normalization, layer.condition_gradient_scale)
    if not isinstance(layer, ConditionedLinear) or layer.variant == 'diagonal':
        return x.new_zeros(*x.shape[:-1], base_layer(layer).out_features)
    if layer.variant == 'dplr':
        return layer.transform.lowrank.delta(x, c)
    if layer.variant == 'basis':
        return layer.transform.delta(x, c)
    if layer.variant == 'static_lora_diagonal':
        return layer.gain * ((x @ layer.U) @ layer.V)
    raise ValueError('Exact full hypernetwork is outside this token pilot')


def rms(value):
    return value.float().square().mean().sqrt().item()


def gradient_alignment(a, b):
    """Zero routes are explicitly undefined; report token as well as global alignment."""
    aa, bb = a.detach().float().flatten(0, -2), b.detach().float().flatten(0, -2)
    na, nb = aa.norm(dim=-1), bb.norm(dim=-1)
    valid = (na > 1e-20) & (nb > 1e-20)
    result = dict(valid_tokens=int(valid.sum().item()), total_tokens=aa.shape[0],
                  cosine=None, mean_token_cosine=None, negative_token_fraction=None)
    if valid.any():
        cos = (aa[valid]*bb[valid]).sum(-1)/(na[valid]*nb[valid])
        result.update(cosine=((aa*bb).sum()/(aa.norm()*bb.norm())).item(),
                      mean_token_cosine=cos.mean().item(), negative_token_fraction=(cos < 0).float().mean().item())
    return result


def decompose_projection_gradient(layer, h, c, upstream, effective_scale=None):
    """Local VJP using the real task-loss cotangent; parameters/.grad are untouched.

    Three independent input leaves separate value, affine-condition and low-rank
    condition paths. Condition Jacobians include normalization. Return raw and
    scaled condition gradients so attenuation is not mistaken for less conflict.
    """
    with torch.enable_grad():
        x = h.detach().requires_grad_()
        ca = c.detach().requires_grad_()
        cl = c.detach().requires_grad_()
        if isinstance(layer, ConditionedLinear):
            if layer.variant != 'dplr':
                raise ValueError('Gradient routing currently supports plain/DPLR')
            ta = condition_input(ca, layer.condition_normalization)
            tl = condition_input(cl, layer.condition_normalization)
            transform = layer.transform
            scaled = x * (1 + transform.to_scale(ta)) + transform.to_shift(ta)
            y = transform.lowrank.base(scaled) + transform.lowrank.delta(x, tl)
            scale = layer.condition_gradient_scale if effective_scale is None else effective_scale
        else:
            y = layer(x); scale = 0.
        grads = torch.autograd.grad(y, (x, ca, cl), upstream.detach(), allow_unused=True)
        gx, ga, gl = (g if g is not None else torch.zeros_like(h) for g in grads)
        return dict(value=gx.detach(), affine_raw=ga.detach(), lowrank_raw=gl.detach(),
                    affine=(ga*scale).detach(), lowrank=(gl*scale).detach(), output=y.detach())


def routing_probe(model, images, labels, label_smoothing=0.):
    """Four fixed validation images; FP32 eager eval-mode task-loss VJPs."""
    records, handles = [], []
    was_training = model.training
    try:
        model.eval()
        with torch.enable_grad():
            for index, block in enumerate(model.blocks):
                def capture(module, args, output, index=index, block=block):
                    h = args[0]; c = args[1] if len(args) > 1 else h
                    records.append((index, block, module, h, c, output))
                handles.append(block.mlp.fc1.register_forward_hook(capture))
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels, label_smoothing=label_smoothing)
            cotangents = torch.autograd.grad(loss, [r[5] for r in records]+[r[3] for r in records])
        for handle in handles: handle.remove()
        handles.clear()
        result = []
        for (index, block, layer, h, c, y), upstream, actual in zip(records, cotangents[:len(records)], cotangents[len(records):]):
            effective = 0. if block.detach_condition else getattr(layer, 'condition_gradient_scale', 1.)
            parts = decompose_projection_gradient(layer, h, c, upstream, effective)
            cond = parts['affine'] + parts['lowrank']; raw_cond = parts['affine_raw'] + parts['lowrank_raw']
            predicted = parts['value'] + cond
            denominator = max(rms(actual), 1e-20)
            relative_error = rms(actual-predicted)/denominator
            if relative_error > 1e-4:
                raise AssertionError(f'Gradient reconstruction failed in block{index}: {relative_error}')
            row = dict(block=index, condition_gradient_scale=effective, input_rms=rms(h),
                       value_gradient_rms=rms(parts['value']), affine_condition_gradient_rms=rms(parts['affine']),
                       lowrank_condition_gradient_rms=rms(parts['lowrank']), condition_gradient_rms=rms(cond),
                       raw_condition_gradient_rms=rms(raw_cond), total_gradient_rms=rms(actual),
                       condition_to_value_ratio=rms(cond)/max(rms(parts['value']),1e-20),
                       reconstruction_relative_error=relative_error,
                       forward_reconstruction_max_error=(parts['output']-y.detach()).abs().max().item(),
                       value_condition_alignment=gradient_alignment(parts['value'], cond),
                       value_raw_condition_alignment=gradient_alignment(parts['value'], raw_cond),
                       value_affine_alignment=gradient_alignment(parts['value'], parts['affine']),
                       value_lowrank_alignment=gradient_alignment(parts['value'], parts['lowrank']),
                       norm_weight_rms=rms(block.norm2.weight), norm_bias_rms=rms(block.norm2.bias))
            if isinstance(layer, ConditionedLinear):
                cc = condition_input(c.detach(), layer.condition_normalization)
                a, b = layer.transform.lowrank.to_coef(cc).chunk(2, -1)
                row.update(condition_rms=rms(cc), coefficient_a_rms=rms(a), coefficient_b_rms=rms(b),
                           condition_token_rms_p95=cc.float().square().mean(-1).sqrt().quantile(.95).item(),
                           coefficient_a_token_rms_p95=a.float().square().mean(-1).sqrt().quantile(.95).item(),
                           coefficient_b_token_rms_p95=b.float().square().mean(-1).sqrt().quantile(.95).item())
            else: row['condition_rms'] = rms(c)
            result.append(row)
        return dict(loss=loss.item(), examples=len(images), blocks=result,
                    definition='Local MLP-input VJPs of smoothed task loss on fixed validation images; raw model, FP32 eval-mode; not training-batch or whole-network conflict')
    finally:
        for handle in handles: handle.remove()
        model.train(was_training)


@contextmanager
def highest_fp32():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


class TokenDiagnostics:
    def __init__(self, model, cfg, validation, device):
        model = getattr(model, '_orig_mod', model)
        self.steps = [int(step) for step in cfg.diagnostic_steps.split(',')]
        self.device = device
        self.routing = cfg.routing_diagnostics
        self.label_smoothing = cfg.label_smoothing
        self.base_hash = parameter_digest(model)
        self.adapter_hash = parameter_digest(model, True)
        generator = torch.Generator().manual_seed(2034)
        probe = torch.randn(2, 5, cfg.width, generator=generator)
        self.probe = torch.nn.functional.layer_norm(probe, (cfg.width,)).to(device)
        self.images = torch.stack([validation[index][0] for index in range(min(8, len(validation)))]).to(device)
        self.labels = (torch.tensor([validation[index][1] for index in range(min(4, len(validation)))], device=device)
                       if self.routing else None)
        self.adapter_ids = hyperparameter_ids(model)
        self.first_gradients = {}
        self.previous = {}
        self.previous_mlp = {}
        self.events = []
        self.hardware = dict(device_name='CPU', device_uuid=None)
        if device.type == 'cuda':
            properties = torch.cuda.get_device_properties(device)
            self.hardware = dict(device_name=properties.name, device_uuid=str(properties.uuid))

    def observe_gradients(self, model, step):
        model = getattr(model, '_orig_mod', model)
        for name, value in model.named_parameters():
            if id(value) in self.adapter_ids and name not in self.first_gradients:
                if value.grad is not None and value.grad.detach().abs().max().item() > 0:
                    self.first_gradients[name] = step

    @torch.no_grad()
    def collect(self, model, step, images=None, labels=None):
        # Hooks/probes run eagerly on shared parameters, outside compiled training.
        model = getattr(model, '_orig_mod', model)
        was_training = model.training
        branches, updates, mlp_updates, groups = [], [], [], {}
        handles = []
        cuda_devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        try:
            model.eval()
            with torch.random.fork_rng(devices=cuda_devices), highest_fp32(), torch.autocast(self.device.type, enabled=False):
                for index, block in enumerate(model.blocks):
                    layer = block.mlp.fc1
                    delta = lowrank_delta(layer, self.probe, self.probe).float().cpu()
                    previous = self.previous.get(index, delta)
                    updates.append(dict(block=index, delta_rms=rms(delta), update_rms=rms(delta-previous),
                                        gap_steps=step-self.events[-1]['step'] if self.events else 0))
                    self.previous[index] = delta.clone()
                    mlp_output = block.mlp(self.probe, self.probe).float().cpu()
                    previous_mlp = self.previous_mlp.get(index, mlp_output)
                    mlp_updates.append(dict(block=index, output_rms=rms(mlp_output),
                                            update_rms=rms(mlp_output-previous_mlp),
                                            gap_steps=step-self.events[-1]['step'] if self.events else 0))
                    self.previous_mlp[index] = mlp_output.clone()
                    if isinstance(layer, ConditionedLinear):
                        group = {}
                        for name, value in layer.named_parameters():
                            if id(value) in self.adapter_ids:
                                group[name] = dict(parameter_rms=rms(value),
                                                  gradient_rms=rms(value.grad) if value.grad is not None else 0.)
                        groups[str(index)] = group

                    def capture(module, args, output, block_index=index):
                        h = args[0]
                        condition = args[1] if len(args) > 1 else h
                        base = base_layer(module)(h) if isinstance(module, ConditionedLinear) else torch.nn.functional.linear(h, module.weight, module.bias)
                        delta_live = lowrank_delta(module, h, condition)
                        total = output-base
                        base_rms = rms(base)
                        branches.append(dict(block=block_index, input_rms=rms(h), base_rms=base_rms,
                                             lowrank_rms=rms(delta_live), total_adapter_rms=rms(total),
                                             lowrank_to_base_rms=rms(delta_live)/max(base_rms,1e-20),
                                             combined_to_base_rms=rms(output)/max(base_rms,1e-20)))
                    handles.append(layer.register_forward_hook(capture))
                model(self.images)
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)
        value = dict(event='token_diagnostic', step=step, model_state='raw',
                     precision='FP32 tensors and highest FP32 matmul precision; no autocast',
                     fixed_lowrank_updates=updates, fixed_mlp_updates=mlp_updates,
                     live_branches=branches, adapter_groups=groups)
        if self.routing:
            with torch.random.fork_rng(devices=cuda_devices), highest_fp32(), torch.autocast(self.device.type, enabled=False):
                value['gradient_routing'] = routing_probe(model, self.images[:4], self.labels, self.label_smoothing)
        if images is not None:
            value.update(batch_images_sha256=tensor_digest(images), batch_labels_sha256=tensor_digest(labels))
        self.events.append(value)
        return value

    def summary(self):
        result = dict(base_initialization_sha256=self.base_hash, adapter_initialization_sha256=self.adapter_hash,
                    probe_seed=2034, probe_sha256=tensor_digest(self.probe), live_images_sha256=tensor_digest(self.images),
                    first_nonzero_gradient=self.first_gradients, hardware=self.hardware,
                    diagnostics=self.events,
                    update_definition='Low-rank delta at fixed independent normalized tokens; each recorded interval, not per-step average.',
                    mlp_update_definition='Complete standalone MLP on the same fixed normalized tokens; interval updates, including base and conditional branches.',
                    limits='Raw-model probes; live eight-image branches; no EMA or entire-network update; plain/diagonal low-rank probes are zero by construction.')
        if self.routing:
            result.update(gradient_probe_images_sha256=tensor_digest(self.images[:4]),
                          gradient_probe_labels_sha256=tensor_digest(self.labels),
                          gradient_probe_label_smoothing=self.label_smoothing)
        return result
