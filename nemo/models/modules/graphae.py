import warnings

import more_itertools
import itertools

import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.nn as gnn

from .utils import NopLayer, select_output_dimensionality_multiplier

class SimpleGraphAEModel(nn.Module):
    def __init__(self, d_in, d_enc = [256,128], d_dec=None):
        super(SimpleGraphAEModel,self).__init__()
        if d_dec is None:
            d_dec = list(reversed(d_enc))
        assert d_enc[-1]==d_dec[0], ValueError("Encoder and decoder must share their last and first dimensionalities")

        self.proj_in = gnn.Linear(d_in, d_enc[0]) if d_in!=d_enc[0] else NopLayer()
        self.proj_out = gnn.Linear(d_dec[-1], d_in) if d_in!=d_enc[0] else NopLayer()
        self.enc = gnn.Sequential(
            "x, edge_index",
            [
                (gnn.GATv2Conv(d_i,d_o),"x, edge_index -> x")
                for d_i, d_o in zip(d_enc[:-1],d_enc[1:])
            ]
        )
        self.dec = gnn.Sequential(
            "x, edge_index",
            [
                (gnn.GATv2Conv(d_i,d_o),"x, edge_index -> x")
                for d_i, d_o in zip(d_dec[:-1],d_dec[1:])
            ]
        )
    
    def forward(self, x:torch.Tensor, edge_index:torch.LongTensor) -> torch.Tensor:
        x_proj_in = self.proj_in(x)
        z = self.enc(x_proj_in, edge_index)
        x_proj_out = self.dec(z, edge_index)
        x_hat = self.proj_out(x_proj_out)
        return x_hat, z


class SimpleMultimodalGraphAEModel(nn.Module):
    def __init__(self, ds_in, d_enc = [256,128], d_dec=None):
        super(SimpleMultimodalGraphAEModel,self).__init__()
        if d_dec is None:
            d_dec = list(reversed(d_enc))
        assert d_enc[-1]==d_dec[0], ValueError("Encoder and decoder must share their last and first dimensionalities")

        self.n_modalities = len(ds_in)
        self.projs_in = nn.ModuleList([
            gnn.Linear(d_in, d_enc[0]) if d_in!=d_enc[0] else NopLayer()
            for d_in in ds_in
        ])
        self.projs_out = nn.ModuleList([
            gnn.Linear(d_dec[-1], d_in) if d_in!=d_enc[0] else NopLayer()
            for d_in in ds_in
        ])

        self.encs = nn.ModuleList([
            gnn.Sequential(
                "x, edge_index",
                [
                    (gnn.GATv2Conv(d_i,d_o),"x, edge_index -> x")
                    for d_i, d_o in zip(d_enc[:-1],d_enc[1:])
                ]
            )
            for _ in range(self.n_modalities)
        ])
        self.decs = nn.ModuleList([
            gnn.Sequential(
                "x, edge_index",
                [
                    (gnn.GATv2Conv(d_i,d_o),"x, edge_index -> x")
                    for d_i, d_o in zip(d_dec[:-1],d_dec[1:])
                ]
            )
            for _ in range(self.n_modalities)
        ])
    
    def forward(self, Xs:list[torch.Tensor], edge_indexes:list[torch.LongTensor]) -> tuple[list[torch.Tensor],list[torch.Tensor],torch.Tensor]:
        zs = []
        for x, edge_index, proj_in, enc in zip(Xs, edge_indexes, self.projs_in, self.encs):
            x_proj_in = proj_in(x)
            zs.append(enc(x_proj_in, edge_index))
        z = torch.sum(torch.stack([zz[:,:,None] for zz in zs],dim=2), dim=2).squeeze(2)
        x_hats = []
        for proj_out, dec in zip(self.projs_out, self.decs):
            x_proj_out = dec(z, edge_index)
            x_hats.append(proj_out(x_proj_out))
        return x_hats, zs, z

try:    
    from scmdc_zinb import ZINBLoss, NBLoss

    class SimpleZINBGraphAEModel(nn.Module):
        def __init__(self, d_in, d_enc = [256,128], d_dec=None):
            super(SimpleZINBGraphAEModel,self).__init__()
            if d_dec is None:
                d_dec = list(reversed(d_enc))
            assert d_enc[-1]==d_dec[0], ValueError("Encoder and decoder must share their last and first dimensionalities")

            self.proj_in = gnn.Linear(d_in, d_enc[0]) if d_in!=d_enc[0] else NopLayer()
            self.proj_out = gnn.Linear(d_dec[-1], d_in*3) if d_in*3!=d_enc[0] else NopLayer()
            self.enc = gnn.Sequential(
                "x, edge_index",
                [
                    (gnn.GATv2Conv(d_i,d_o),"x, edge_index -> x")
                    for d_i, d_o in zip(d_enc[:-1],d_enc[1:])
                ]
            )
            self.dec = gnn.Sequential(
                "x, edge_index",
                [
                    (gnn.GATv2Conv(d_i,d_o),"x, edge_index -> x")
                    for d_i, d_o in zip(d_dec[:-1],d_dec[1:])
                ]
            )
        
        def forward(self, x:torch.Tensor, edge_index:torch.LongTensor) -> torch.Tensor:
            x_proj_in = self.proj_in(x)
            z = self.enc(x_proj_in, edge_index)
            x_proj_out = self.dec(z, edge_index)
            x_hat = self.proj_out(x_proj_out)
            return x_hat, z

except ImportError:
    warnings.warn("Failed to import ZINBLoss from scmdc_zinb, ZINB-based models will not be available or won't have full functionality")

class SimpleMultimodalMultilossGraphAEModel(nn.Module):
    def __init__(self, ds_in, modalities_tgt_dist=None, d_enc = [256,128], d_dec=None):
        super(SimpleMultimodalMultilossGraphAEModel,self).__init__()
        if d_dec is None:
            d_dec = list(reversed(d_enc))
        assert d_enc[-1]==d_dec[0], ValueError("Encoder and decoder must share their last and first dimensionalities")

        self.n_modalities = len(ds_in)
        self.modalities_tgt_dist = modalities_tgt_dist
        if self.modalities_tgt_dist is None:
            self.modalities_tgt_dist = [nn.MSELoss() for _ in range(self.n_modalities)]

        self.projs_in = nn.ModuleList([
            gnn.Linear(d_in, d_enc[0]) if d_in!=d_enc[0] else NopLayer()
            for d_in in ds_in
        ])

        self.projs_out = nn.ModuleList([
            gnn.Linear(d_dec[-1], d_in*select_output_dimensionality_multiplier(m)) if d_in*select_output_dimensionality_multiplier(m)!=d_enc[0] else NopLayer()
            for d_in, m in zip(ds_in, self.modalities_tgt_dist)
        ])

        self.encs = nn.ModuleList([
            gnn.Sequential(
                "x, edge_index",
                [
                    (gnn.GATv2Conv(d_i,d_o),"x, edge_index -> x")
                    for d_i, d_o in zip(d_enc[:-1],d_enc[1:])
                ]
            )
            for _ in range(self.n_modalities)
        ])
        self.decs = nn.ModuleList([
            gnn.Sequential(
                "x, edge_index",
                [
                    (gnn.GATv2Conv(d_i,d_o),"x, edge_index -> x")
                    for d_i, d_o in zip(d_dec[:-1],d_dec[1:])
                ]
            )
            for _ in range(self.n_modalities)
        ])
    
    def forward(self, Xs:list[torch.Tensor], edge_indexes:list[torch.LongTensor]) -> tuple[list[torch.Tensor],list[torch.Tensor],torch.Tensor]:
        zs = []
        for x, edge_index, proj_in, enc in zip(Xs, edge_indexes, self.projs_in, self.encs):
            x_proj_in = proj_in(x)
            zs.append(enc(x_proj_in, edge_index))
        z = torch.sum(torch.stack([zz[:,:,None] for zz in zs],dim=2), dim=2).squeeze(2)
        x_hats = []
        for edge_index, proj_out, dec in zip(edge_indexes, self.projs_out, self.decs):
            x_proj_out = dec(z, edge_index)
            x_hats.append(proj_out(x_proj_out))
        return x_hats, zs, z