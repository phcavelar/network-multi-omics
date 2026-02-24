import more_itertools
import itertools

import torch
import torch.nn as nn

from .utils import select_output_dimensionality_multiplier, NopLayer

class SimpleAEModel(nn.Module):
    def __init__(self, d_in, d_enc = [256,128], d_dec=None, Act=nn.ReLU, act_args=[], act_kwargs={}):
        super(SimpleAEModel,self).__init__()
        if d_dec is None:
            d_dec = list(reversed(d_enc))
        assert d_enc[-1]==d_dec[0], ValueError("Encoder and decoder must share their last and first dimensionalities")

        self.ds = [d_in, *d_enc]

        self.enc = nn.Sequential(
            *more_itertools.interleave(
                [
                    nn.Linear(d_i,d_o)
                    for d_i, d_o in zip(self.ds[:-1], self.ds[1:])
                ],
                itertools.repeat(Act(*act_args,**act_kwargs))
            )
        )
        self.dec = nn.Sequential(
            *more_itertools.interleave(
                [
                    nn.Linear(d_i,d_o)
                    for d_i, d_o in zip(self.ds[-1:0:-1], self.ds[-2::-1]) #reversed
                ],
                itertools.repeat(Act(*act_args,**act_kwargs))
            )
        )
    
    def forward(self, x:torch.Tensor, edge_index:torch.LongTensor=None) -> torch.Tensor:
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat, z
    
class SimpleMultimodalMultilossAEModel(nn.Module):
    def __init__(self, ds_in, modalities_tgt_dist=None, d_enc = [256,128], d_dec=None, Act=nn.ReLU, act_args=[], act_kwargs={}):
        super(SimpleMultimodalMultilossAEModel,self).__init__()
        if d_dec is None:
            d_dec = list(reversed(d_enc))
        assert d_enc[-1]==d_dec[0], ValueError("Encoder and decoder must share their last and first dimensionalities")

        self.n_modalities = len(ds_in)
        self.modalities_tgt_dist = modalities_tgt_dist
        if self.modalities_tgt_dist is None:
            self.modalities_tgt_dist = [nn.MSELoss() for _ in range(self.n_modalities)]

        self.projs_in = nn.ModuleList([
            nn.Linear(d_in, d_enc[0]) if d_in!=d_enc[0] else NopLayer()
            for d_in in ds_in
        ])

        self.projs_out = nn.ModuleList([
            nn.Linear(d_dec[-1], d_in*select_output_dimensionality_multiplier(m)) if d_in*select_output_dimensionality_multiplier(m)!=d_enc[0] else NopLayer()
            for d_in, m in zip(ds_in, self.modalities_tgt_dist)
        ])

        self.encs = nn.ModuleList([
                nn.Sequential(
                *more_itertools.interleave(
                    [
                        nn.Linear(d_i,d_o)
                        for d_i, d_o in zip(self.ds[:-1], self.ds[1:])
                    ],
                    itertools.repeat(Act(*act_args,**act_kwargs))
                )
            )
            for _ in range(self.n_modalities)
        ])
        self.decs = nn.ModuleList([
            nn.Sequential(
                *more_itertools.interleave(
                    [
                        nn.Linear(d_i,d_o)
                        for d_i, d_o in zip(self.ds[-1:0:-1], self.ds[-2::-1]) #reversed
                    ],
                    itertools.repeat(Act(*act_args,**act_kwargs))
                )
            )
            for _ in range(self.n_modalities)
        ])
    
    def forward(self, Xs:list[torch.Tensor], edge_indexes:list[torch.LongTensor]=None) -> tuple[list[torch.Tensor],list[torch.Tensor],torch.Tensor]:
        zs = []
        for x, proj_in, enc in zip(Xs, self.projs_in, self.encs):
            x_proj_in = proj_in(x)
            zs.append(enc(x_proj_in))
        z = torch.sum(torch.stack([zz[:,:,None] for zz in zs],dim=2), dim=2).squeeze(2)
        x_hats = []
        for proj_out, dec in zip(self.projs_out, self.decs):
            x_proj_out = dec(z)
            x_hats.append(proj_out(x_proj_out))
        return x_hats, zs, z