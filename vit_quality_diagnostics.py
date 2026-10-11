"""Read-only raw-model probes for token-conditioned ViT training."""

from contextlib import contextmanager
import hashlib

import torch

from conditioning import ConditionedLinear
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
        self.base_hash = parameter_digest(model)
        self.adapter_hash = parameter_digest(model, True)
        generator = torch.Generator().manual_seed(2034)
        probe = torch.randn(2, 5, cfg.width, generator=generator)
        self.probe = torch.nn.functional.layer_norm(probe, (cfg.width,)).to(device)
        self.images = torch.stack([validation[index][0] for index in range(min(8, len(validation)))]).to(device)
        self.adapter_ids = hyperparameter_ids(model)
        self.first_gradients = {}
        self.previous = {}
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
        branches, updates, groups = [], [], {}
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
                     fixed_lowrank_updates=updates, live_branches=branches, adapter_groups=groups)
        if images is not None:
            value.update(batch_images_sha256=tensor_digest(images), batch_labels_sha256=tensor_digest(labels))
        self.events.append(value)
        return value

    def summary(self):
        return dict(base_initialization_sha256=self.base_hash, adapter_initialization_sha256=self.adapter_hash,
                    probe_seed=2034, probe_sha256=tensor_digest(self.probe), live_images_sha256=tensor_digest(self.images),
                    first_nonzero_gradient=self.first_gradients, hardware=self.hardware,
                    diagnostics=self.events,
                    update_definition='Low-rank delta at fixed independent normalized tokens; each recorded interval, not per-step average.',
                    limits='Raw-model probes; live eight-image branches; no EMA or entire-network update; plain/diagonal low-rank probes are zero by construction.')
