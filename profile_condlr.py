#!/usr/bin/env python3
"""Bounded layer profiling; writes JSON evidence, never model weights."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from conditioning import ConditionedLinear, VARIANTS
from condlr import NaiveCondLowRank


def profile(variant, rank, chunk, args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        # Compare allocator peaks from each workload rather than inherited cache.
        torch.cuda.empty_cache()
    base = torch.nn.Linear(args.width, int(args.width * args.mlp_ratio))
    layer = base if variant == "plain" else ConditionedLinear(base, "exact" if variant == "naive" else variant,
                                                              rank, args.n_basis, chunk)
    layer = layer.to(device)
    if variant == "naive":
        layer = NaiveCondLowRank(layer.transform)
    # Nonzero updates exercise both generator sides and avoid a vacuous oracle.
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if name.endswith((".B", ".V")):
                p.normal_(std=0.005)
    x = torch.randn(1, args.tokens, args.width, device=device, requires_grad=True)
    c = torch.randn_like(x, requires_grad=True)
    params = list(layer.parameters())
    parameter_storages = {p.untyped_storage().data_ptr() for p in params}
    storages = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        if pointer not in parameter_storages:
            storages[pointer] = storage.nbytes()
        return tensor

    def forward():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            return layer(x) if variant == "plain" else layer(x, c)

    def step():
        layer.zero_grad(set_to_none=True)
        x.grad, c.grad = None, None
        loss = forward().float().square().mean()
        loss.backward()

    # Autocast weight copies are retained by ordinary Linear autograd. These
    # count as non-parameter saved storage; label that inclusion explicitly.
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = forward()
    saved_bytes = sum(storages.values())
    del output
    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {"variant": variant, "rank": rank, "chunk": chunk,
            "parameters": sum(p.numel() for p in params), "saved_nonparameter_storage_bytes": saved_bytes,
            "milliseconds_per_forward_backward": elapsed * 1000 / args.steps,
            "train_tokens_per_second": args.tokens * args.steps / elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--chunks", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--n-basis", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/layer-profile.json"))
    args = parser.parse_args()
    if min(args.width, args.tokens, args.n_basis, args.steps, *args.ranks, *args.chunks) < 1 or args.warmup < 0:
        parser.error("dimensions, ranks, chunks and steps must be positive; warmup nonnegative")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.output.exists():
        raise FileExistsError(args.output)
    records = []
    for rank in args.ranks:
        for variant in (*VARIANTS, "naive"):
            for chunk in args.chunks if variant == "exact" else [args.chunks[-1]]:
                record = profile(variant, rank, chunk, args)
                records.append(record)
                print(json.dumps(record), flush=True)
    from train_cifar10 import provenance
    payload = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "provenance": provenance(),
               "saved_storage_note": "unique backing storages, original parameter storage excluded; includes autocast weight copies",
               "records": records}
    payload["provenance"]["source_sha256"]["profile_condlr.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
