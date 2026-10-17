"""Condition-generated TT matrices applied by mode contractions, never densified.

Channel coordinates are tensorized in the given mode order. TT bond rank is
not the ordinary matrix rank. The static TT construction follows Novikov et al.
(2015); condition-generated cores are this project's adaptation.
"""
import math

import torch
from torch import nn


def tt_spec(width, modes, bond_rank):
    if isinstance(modes, str):
        try:
            modes = tuple(int(v) for v in modes.split(','))
        except ValueError as error:
            raise ValueError('TT modes must be comma-separated positive integers') from error
    else:
        modes = tuple(modes)
    if (len(modes) < 2 or any(type(m) is not int or m < 2 for m in modes)
            or math.prod(modes) != width or type(bond_rank) is not int or bond_rank < 1):
        raise ValueError('TT requires at least two modes >=2, product equal to width, and positive integer bond rank')
    ranks = (1,) + (bond_rank,) * (len(modes)-1) + (1,)
    return modes, tuple((ranks[i], m, m, ranks[i+1]) for i,m in enumerate(modes))


def tt_apply(x, cores):
    """x=[B,T,prod(m)], core_i=[B,r_prev,m_i,n_i,r_next].

    State after each contraction: [B,T,output_prefix,input_suffix,bond].
    For square modes its size is B*T*d*bond, not B*d*d. No dense matrix,
    token-dependent weights, or full Kronecker product is constructed.
    """
    b,t,d = x.shape
    state = x.to(cores[0].dtype).reshape(b,t,1,d,1)
    prefix, suffix = 1,d
    for core in cores:
        _,left,m,n,right = core.shape
        suffix //= m
        state = state.reshape(b,t,prefix,m,suffix,left)
        state = torch.einsum('btpisa,baijz->btpjsz', state, core)
        prefix *= n
        state = state.reshape(b,t,prefix,suffix,right)
    return state.reshape(b,t,prefix)


class ConditionalTT(nn.Module):
    is_square = True
    variant = 'dplr'
    kind = 'tt'

    def __init__(self, width, condition_width, n_basis, modes, bond_rank, gain):
        super().__init__()
        self.modes, self.core_shapes = tt_spec(width,modes,bond_rank)
        if min(condition_width,n_basis)<1 or not math.isfinite(gain) or gain<=0:
            raise ValueError('Invalid conditional TT dimensions or gain')
        self.d_in = self.d_out = self.rank = width  # Ordinary matrix rank bound.
        self.d_condition, self.n_basis = condition_width,n_basis
        self.bond_rank, self.gain, self.factor_offsets = bond_rank,float(gain),True
        self.core_sizes = tuple(math.prod(s) for s in self.core_shapes)
        self.coefficients = nn.Linear(condition_width,n_basis)
        # One shared affine latent, separate affine core decoders with offsets.
        self.core_bases = nn.ParameterList([nn.Parameter(torch.zeros(n_basis,size)) for size in self.core_sizes])
        self.core_offsets = nn.ParameterList([nn.Parameter(torch.zeros(shape)) for shape in self.core_shapes])
        # A single zero end core makes the branch exactly zero. Earlier cores
        # have nonzero static offsets; all generators can wake after it opens.
        # Fan-in convention: independent nonzero cores have E[||xW||²]=E[||x||²]
        # when *all* cores have this variance. The zero last core intentionally
        # breaks this at initialization; no trained variance/muP guarantee.
        for offset,shape in zip(self.core_offsets[:-1], self.core_shapes[:-1]):
            nn.init.normal_(offset,std=1/math.sqrt(shape[0]*shape[1]))

    def decoder_named_parameters(self):
        return [(f'core_bases.{i}',p) for i,p in enumerate(self.core_bases)] + [(f'core_offsets.{i}',p) for i,p in enumerate(self.core_offsets)]

    def cores(self, condition):
        if condition.ndim!=2 or condition.shape[-1]!=self.d_condition:
            raise ValueError('Expected broadcast condition [B,q]')
        latent = self.coefficients(condition)
        result=[]
        for basis,offset,shape in zip(self.core_bases,self.core_offsets,self.core_shapes):
            value = (latent @ basis).reshape(condition.shape[0],*shape)
            result.append(value + offset.to(value.dtype))
        return result

    def forward(self,x,condition,factors=None):
        if x.ndim!=3 or x.shape[0]!=condition.shape[0] or x.shape[-1]!=self.d_in:
            raise ValueError('Expected x=[B,T,d] and condition=[B,q]')
        if factors is not None:
            raise ValueError('Shared TT factor caching is not supported')
        return tt_apply(x,self.cores(condition))*self.gain

    def forward_matmul_flops(self,tokens):
        generation=2*self.d_condition*self.n_basis+2*self.n_basis*sum(self.core_sizes)
        contractions=sum(2*tokens*self.d_in*m*left*right for left,m,_,right in self.core_shapes)
        return generation+contractions

    @torch.no_grad()
    def operator_diagnostics(self,condition):
        identity=torch.eye(self.d_in,device=condition.device,dtype=condition.dtype)
        matrix=self(identity[None].expand(condition.shape[0],-1,-1),condition)
        return dict(operator='tt',tt_modes=list(self.modes),tt_bond_rank=self.bond_rank,
                    transform_frobenius_mean=matrix.float().square().sum((-2,-1)).sqrt().mean().item(),
                    parameter_rms={n:p.float().square().mean().sqrt().item() for n,p in self.decoder_named_parameters()},
                    generated_core_rms=[v.float().square().mean().sqrt().item() for v in self.cores(condition)],
                    matrix_materialization_scope='untimed diagnostic only; square feature units')
