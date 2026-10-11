#!/usr/bin/env python3
"""Generator-budget and conditional-core cases for the shared engineering runner."""

import math

from benchmark_dit_design import main


def generator_cases(width=192):
    def case(name, variant="dplr", rank=8, n_basis=16, mlp_ratio=4., core_rank=2, raw=False):
        return dict(name=name, variant=variant, rank=rank, n_basis=n_basis,
                    mlp_ratio=mlp_ratio, core_rank=core_rank, adapter_input="normalized",
                    adapter_gain=1. if raw or variant == "adaln_zero" else 1 / math.sqrt(rank))

    return [case("adaln", "adaln_zero"),
            case("ffn_r2_budget", "adaln_zero", mlp_ratio=4.5),
            case("basis_r2_s16", rank=2),
            case("basis_r8_s16"), case("basis_r8_s16_raw", raw=True),
            *[case(f"basis_r8_q{divisor}", n_basis=max(1, width // divisor)) for divisor in (4, 16, 32)],
            case("basis_r16_q32", rank=16, n_basis=max(1, width // 32)),
            case("bottleneck_r8", "conditional_bottleneck"),
            *[case(f"core_r{rank}", "conditional_core", rank=rank) for rank in (4, 8, 16)],
            case("dplr_core_r8_p2", "conditional_dplr_core", core_rank=2)]


if __name__ == "__main__":
    main()
