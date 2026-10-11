"""Frozen auxiliary classifier agreement; a diagnostic, not ground-truth fidelity."""

import hashlib
import json
import importlib.util
from pathlib import Path
import sys

import torch

def frozen_classifier_model(directory):
    """Load the evaluator's verified original implementation, isolated from training.

    Frozen research runners use absolute local imports. Temporarily bind those
    four dependencies while loading under unique module names, then restore the
    active trainer modules. The returned model retains its original globals.
    """
    directory = Path(directory)
    source = json.loads((directory / 'provenance.json').read_text())['source_sha256']
    names = ('condlr', 'conditioning', 'vit', 'train_cifar10')
    if set(source) != {name + '.py' for name in names}:
        raise ValueError('Unexpected frozen classifier source manifest')
    candidates = [parent / 'runner' for parent in directory.parents]
    runner = next((path for path in candidates if all(
        (path / name).is_file() and hashlib.sha256((path / name).read_bytes()).hexdigest() == value
        for name, value in source.items())), None)
    if runner is None:
        raise ValueError('Verified original classifier runner is unavailable')
    namespace = '_adalora_classifier_' + hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            unique = namespace + '_' + name
            module = sys.modules.get(unique)
            if module is None:
                spec = importlib.util.spec_from_file_location(unique, runner / (name + '.py'))
                module = importlib.util.module_from_spec(spec)
                sys.modules[unique] = module
                try:
                    spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(unique, None)
                    raise
            sys.modules[name] = module
        cfg = module.Config(**json.loads((directory / 'config.json').read_text()))
        return cfg, module.build_model(cfg)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def classifier_provenance(directory):
    directory = Path(directory)
    return {"run": str(directory.resolve()),
            "sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                       for name in ("model.pt", "config.json", "dataset.json", "provenance.json", "results.json")},
            "weight_bytes": (directory / "model.pt").stat().st_size,
            "purpose": "fixed auxiliary evaluator of generated requested-class agreement",
            "retention": "one frozen evaluator and its compact training records until explicitly removed"}


class ClassAgreement:
    def __init__(self, model, device, metadata=None):
        self.model = model.to(device).requires_grad_(False).eval()
        self.device, self.metadata = device, metadata
        self.mean = torch.tensor((0.4914, 0.4822, 0.4465), device=device)[None, :, None, None]
        self.std = torch.tensor((0.2470, 0.2435, 0.2616), device=device)[None, :, None, None]
        self.matrices = {key: torch.zeros((10, 10), device=device, dtype=torch.int64) for key in ("real", "generated")}

    @classmethod
    def from_run(cls, directory, device, validation_indices):
        directory = Path(directory)
        cfg, model = frozen_classifier_model(directory)
        dataset = json.loads((directory / "dataset.json").read_text())
        if cfg.synthetic or cfg.evaluate_test or cfg.train_limit is not None or cfg.val_limit is not None:
            raise ValueError("Auxiliary classifier must use only the full real train/validation split")
        if dataset["validation_indices"] != list(validation_indices):
            raise ValueError("Classifier and diffusion validation indices differ")
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True), strict=True)
        return cls(model, device, classifier_provenance(directory))

    @torch.no_grad()
    def update(self, images, labels, key):
        if key not in self.matrices or images.ndim != 4 or images.shape[1:] != (3, 32, 32):
            raise ValueError("Expected RGB32 images and a real/generated metric key")
        images, labels = images.to(self.device).float(), labels.to(self.device)
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Auxiliary classifier expects finite RGB in [0,1]")
        if labels.shape != (len(images),) or labels.min() < 0 or labels.max() >= 10:
            raise ValueError("Invalid classifier labels")
        predicted = self.model((images - self.mean) / self.std).argmax(dim=-1)
        self.matrices[key] += torch.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)

    def compute(self):
        result = {"classifier": self.metadata, "input": "unquantized RGB32 [0,1], CIFAR mean/std, FP32 classifier",
                  "interpretation": "Auxiliary classifier agreement, not human class accuracy; limited by classifier errors and domain shift.",
                  "uniform_chance_agreement": 0.1}
        for key, matrix in self.matrices.items():
            counts = matrix.sum(1)
            if matrix.sum() == 0:
                raise ValueError("Both real and generated denominators are required")
            result[key] = {"examples": matrix.sum().item(), "agreement": (matrix.diag().sum() / matrix.sum()).item(),
                           "class_counts": counts.tolist(), "predicted_class_counts": matrix.sum(0).tolist(),
                           "per_class_agreement": [a / b if b else None for a, b in zip(matrix.diag().tolist(), counts.tolist())],
                           "confusion_rows_requested_columns_predicted": matrix.tolist()}
        return result
