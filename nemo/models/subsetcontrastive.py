import inspect
import itertools
import warnings
import json
import os
import os.path as osp
from typing import Literal, Optional, Union
import time

import tqdm

import numpy as np
import scanpy as sc

from sklearn.base import TransformerMixin, ClusterMixin

import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.nn as gnn
from .modules.graphae import SimpleMultimodalMultilossGraphAEModel
from .modules.utils import select_output_dimensionality_multiplier, ZINBLoss, NBLoss

class SubsetContrastive(TransformerMixin, ClusterMixin):

    def __init__(
            self,
            adatas:dict[str,sc.AnnData] = None,
            criteria:dict[str,nn.modules.loss._Loss] = None,
            sample_size:Union[float,int] = None,
            min_overlap = 0.1,
            d_enc = None,
            lr = 1e-4,
            num_epochs = 128,
            t_contrastive = 0,
            reconstruction_beta = 10.,
            contrastive_alpha = 1.,
            n_neighbors = 15,
            path_to_load_from=None,
            version_to_load_as=None,
            force_disable_cuda=True,
            only_convert_subsets_to_torch=True,
            **model_kwargs,
            ) -> None:
        if path_to_load_from is not None:
            self.load(path_to_load_from,version_to_load_as)
            return
        if any((adatas is None, criteria is None, sample_size is None)):
            raise ValueError("Must pass adatas, criteria and sample_size parameters if not loading from a file!")

        if d_enc is None:
            self.d_enc = inspect.signature(SimpleMultimodalMultilossGraphAEModel.__init__).parameters["d_enc"].default
        else:
            self.d_enc = d_enc
        
        self.sample_size = sample_size
        self.min_overlap = min_overlap
        self.num_epochs = num_epochs
        self.t_contrastive = t_contrastive
        self.reconstruction_beta = reconstruction_beta
        self.contrastive_alpha = contrastive_alpha
        self.n_neighbors = n_neighbors

        self.modalities = sorted(adatas.keys())
        self.mod_criteria = [criteria[m] for m in self.modalities]
        self.model = SimpleMultimodalMultilossGraphAEModel(
            ds_in=[adatas[m].X.shape[1] for m in self.modalities],
            modalities_tgt_dist=self.mod_criteria,
            d_enc=self.d_enc,   
        )

        self.contrastive_modules = [
            nn.Bilinear(self.d_enc[-1],self.d_enc[-1],1)
            for _ in self.modalities
        ]

        self.opt = torch.optim.Adam(
            itertools.chain(
                self.model.parameters(),
                *[con_mod.parameters() for con_mod in self.contrastive_modules],
            ),
            lr=lr)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt)
        self.force_disable_cuda = force_disable_cuda
        self.only_convert_subsets_to_torch = only_convert_subsets_to_torch

    def train(
            self,
            adatas:Union[dict[str,sc.AnnData]],
            disable_tqdm = False,
            ):
        n_samples = adatas[self.modalities[0]].shape[0]
        sample_size = int(
            self.sample_size
            if isinstance(self.sample_size, int) else
            self.sample_size*n_samples
        )

        if self.only_convert_subsets_to_torch:
            Xs = [adatas[m].X for m in self.modalities]
        else:
            Xs = [torch.tensor(adatas[m].X) for m in self.modalities]

        if not self.force_disable_cuda and torch.cuda.is_available():
            print("Using CUDA")
            if not self.only_convert_subsets_to_torch:
                Xs = [X.cuda() for X in Xs]
            #edge_index = [edge_index.cuda() for edge_index in edge_indexes]
            self.model = self.model.cuda()
        
        cur_device = self.model.projs_in[0].weight.device

        losses = {
            "total": [],
            "r": [],
            **{f"r_{i}": [] for i in range(len(Xs))},
            "c": [],
            "c1": [],
            **{f"c1_{i}": [] for i in range(len(Xs))},
            "c2": [],
            **{f"c2_{i}": [] for i in range(len(Xs))},
            "|∇|": [],
            "t_ss": [],
            "t_fw": [],
            "t_cl": [],
            "t_bw": [],
        }

        contrastive_target_factory = lambda pos, total: torch.concatenate([torch.ones(pos),torch.zeros(total-pos)])

        tqdm_iter = tqdm.tqdm(range(self.num_epochs), disable=disable_tqdm)
        for e in tqdm_iter:
            if len(losses["total"])>0:
                tqdm_iter.set_description(
                    f"{losses['total'][-1]:.4f}, {losses['r'][-1]:.4f}, {losses['c1'][-1]:.4f}, {losses['c2'][-1]:.4f}, {losses['|∇|'][-1]:.4f}"
                )
            t_es = time.time()

            with torch.no_grad():
                while True:
                    idx_1:np.ndarray = np.random.choice(n_samples, size=sample_size, replace=False)
                    idx_2:np.ndarray = np.random.choice(n_samples, size=sample_size, replace=False)
                    overlap = set(idx_1).intersection(set(idx_2))
                    if len(overlap)/sample_size<self.min_overlap:
                        continue
                    idx_o = np.asarray(list(overlap))
                    idx_o1, idx_o2 = [], []
                    for o in idx_o:
                        idx_o1.append(np.where(idx_1==o)[0].item())
                        idx_o2.append(np.where(idx_2==o)[0].item())
                    idx_o1, idx_o2 = map(np.asarray, (idx_o1, idx_o2))
                    msk_o1, msk_o2 = np.zeros(sample_size, dtype=bool), np.zeros(sample_size, dtype=bool)
                    msk_o1[idx_o1] = 1
                    msk_o2[idx_o2] = 1

                    As_1, As_2 = [], []
                    for m in self.modalities:
                        ad1 = sc.AnnData(adatas[m].X[idx_1])
                        ad2 = sc.AnnData(adatas[m].X[idx_2])
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            sc.pp.neighbors(ad1, n_neighbors=self.n_neighbors)
                            sc.pp.neighbors(ad2, n_neighbors=self.n_neighbors)
                        As_1.append(torch.tensor(np.stack(ad1.obsp["connectivities"].nonzero()), dtype=torch.long, device=cur_device))
                        As_2.append(torch.tensor(np.stack(ad2.obsp["connectivities"].nonzero()), dtype=torch.long, device=cur_device))
                        del ad1
                        del ad2
                    break
            
            if self.only_convert_subsets_to_torch:
                Xs_1 = [torch.tensor(X[idx_1], device=cur_device) for X in Xs]
                Xs_2 = [torch.tensor(X[idx_2], device=cur_device) for X in Xs]
            else:
                Xs_1 = [X[idx_1] for X in Xs]
                Xs_2 = [X[idx_2] for X in Xs]

            self.opt.zero_grad()
            t_pp = time.time()
            x_hats_1, zs_1, z_1 = self.model(Xs_1, As_1)
            x_hats_2, zs_2, z_2 = self.model(Xs_2, As_2)
            t_fw = time.time()
            idb = x_hats_1[0].shape[1]//3

            # Contrastive loss
            N_zs_1 = [
                gnn.avg_pool_neighbor_x(pyg.data.Data(z,edge_index), flow="source_to_target").x
                for z, edge_index in zip(zs_1, As_1)
            ]
            N_zs_2 = [
                gnn.avg_pool_neighbor_x(pyg.data.Data(z,edge_index), flow="source_to_target").x
                for z, edge_index in zip(zs_2, As_2)
            ]
            c1s = [
                torch.cat(
                    [
                        con_mod(N_z1[idx_o1], N_z2[idx_o2]),
                        con_mod(N_z1[~msk_o1], N_z2[~msk_o2]),
                    ]
                )
                for con_mod, N_z1, N_z2 in zip(self.contrastive_modules, N_zs_1, N_zs_2)
            ]
            contrastive_targets = contrastive_target_factory(idx_o.shape[0], sample_size)
            lc1s = [nn.functional.binary_cross_entropy_with_logits(c1.squeeze(),contrastive_targets) for c1 in c1s]
            lc2s = [
                torch.tensor(0)
                #nn.functional.binary_cross_entropy_with_logits(c2.squeeze(),contrastive_targets)
                for _ in c1s
            ]
            lrs = []
            for criterion, xh1, xh2, X1, X2 in zip(self.mod_criteria, x_hats_1, x_hats_2, Xs_1, Xs_2):
                n_out = select_output_dimensionality_multiplier(criterion)
                x_hat = torch.concatenate([xh1,xh2])
                X = torch.concatenate([X1,X2])
                if isinstance(criterion, (nn.MSELoss,nn.BCEWithLogitsLoss,nn.BCELoss)):
                    this_lr = criterion(x_hat, X)
                elif isinstance(criterion, NBLoss):
                    x_mu, x_disp = x_hat[:,0*idb:1*idb], x_hat[:,1*idb:2*idb]
                    x_mu = torch.clamp(torch.exp(x_mu), min=1e-5, max=1e6)
                    x_disp = torch.clamp(nn.functional.softplus(x_disp), min=1e-5, max=1e6)
                    this_lr = criterion(X, x_mu, x_disp)
                elif isinstance(criterion, ZINBLoss):
                    x_mu, x_disp, x_pi = x_hat[:,0*idb:1*idb], x_hat[:,1*idb:2*idb], x_hat[:,2*idb:3*idb]
                    x_mu = torch.clamp(torch.exp(x_mu), min=1e-5, max=1e6)
                    x_disp = torch.clamp(nn.functional.softplus(x_disp), min=1e-5, max=1e6)
                    x_pi = torch.sigmoid(x_pi)
                    this_lr = criterion(X, x_mu, x_disp, x_pi)
                else:
                    raise ValueError(f"Invalid target distribution {criterion}")
                    idb = x_hat.shape[1]//n_out
                lrs.append(this_lr)
            
            lr = sum(lrs)
            lc1 = sum(lc1s)
            lc2 = sum(lc2s)
            if e>=self.t_contrastive:
                l = self.reconstruction_beta*lr + self.contrastive_alpha*(lc1+lc2)
            else:
                l = lr
            t_cl = time.time()
            l.backward()
            
            #grad_norm = torch.tensor(0)
            grad_norm = nn.utils.clip_grad_norm_(
                itertools.chain(
                    self.model.parameters(),
                    *[con_mod.parameters() for con_mod in self.contrastive_modules],
                ),
                1,
                error_if_nonfinite=True,
            )
            self.opt.step()
            self.sched.step(lr)
            t_bw = time.time()
            with torch.no_grad():
                losses["total"].append(l.detach().cpu().numpy().item())
                losses["r"].append(lr.detach().cpu().numpy().item())
                losses["c1"].append(lc1.detach().cpu().numpy().item())
                losses["c2"].append(lc2.detach().cpu().numpy().item())
                losses["c"].append((lc1+lc2).detach().cpu().numpy().item())
                for i in range(len(lrs)):
                    losses[f"r_{i}"].append(lrs[i].detach().cpu().numpy().item())
                    losses[f"c1_{i}"].append(lc1s[i].detach().cpu().numpy().item())
                    losses[f"c2_{i}"].append(lc2s[i].detach().cpu().numpy().item())
                losses["|∇|"].append(grad_norm.detach().cpu().numpy().item())
                losses["t_ss"].append(t_pp-t_es)
                losses["t_fw"].append(t_fw-t_pp)
                losses["t_cl"].append(t_cl-t_fw)
                losses["t_bw"].append(t_bw-t_cl)
        
        self.history = losses

        return losses
    
    def save(self,path):
        self.__save_v1(path)

    def load(self,path, v):
        match v:
            case 1 | _:
                self.__load_v1(path)
    
    def __save_v1(self,path):
        loading_dict = dict(
            adatas = None,
            criteria = None,
            sample_size = self.sample_size,
            min_overlap = self.min_overlap,
            d_enc = self.d_enc,
            lr = self.opt.state_dict()["param_groups"][0]["lr"],
            num_epochs = self.num_epochs,
            t_contrastive = self.t_contrastive,
            reconstruction_beta = self.reconstruction_beta,
            contrastive_alpha = self.contrastive_alpha,
            ds_in = [l.weight.shape[1] for l in self.model.projs_in],
            modalities = self.modalities,
            len_mod_criteria = len(self.mod_criteria),
            len_contrastive_modules = len(self.contrastive_modules),
        )
        os.makedirs(path, exist_ok=True) #FIXME change to false
        with open(osp.join(path,"metadata.json"),"w") as f:
            json.dump({"version": 1},f)
        with open(osp.join(path,"loading_dict.json"), "w") as f:
            json.dump(loading_dict,f)
        for i in range(len(self.mod_criteria)):
            torch.save(
                self.mod_criteria[i],
                osp.join(path,f"mod_criteria_{i}")
            )
        torch.save(
            self.model.state_dict(),
            osp.join(path,"model")
        )
        for i in range(len(self.contrastive_modules)):
            torch.save(
                self.contrastive_modules[i].state_dict(),
                osp.join(path,f"contrastive_modules_{i}")
            )
        torch.save(
            self.opt.state_dict(),
            osp.join(path,"opt")
        )
        torch.save(
            self.sched.state_dict(),
            osp.join(path,"sched")
        )
        if hasattr(self,"history"):
            with open(osp.join(path,"history.json"), "w") as f:
                json.dump(self.history,f)
    
    def __load_v1(self,path):
        with open(osp.join(path,"metadata.json"), "r") as f:
            metadata = json.load(f)
        assert metadata["version"] == 1, f"Trying to load a model saved with version {metadata['version']} with load_v1"

        with open(osp.join(path,"loading_dict.json"), "r") as f:
            loading_dict = json.load(f)

        lr = loading_dict["lr"]
        ds_in = loading_dict["ds_in"]
        len_mod_criteria = loading_dict["len_mod_criteria"]
        len_contrastive_modules = loading_dict["len_contrastive_modules"]

        self.d_enc = loading_dict["d_enc"]
        self.sample_size = loading_dict["sample_size"]
        self.min_overlap = loading_dict["min_overlap"]
        self.num_epochs = loading_dict["num_epochs"]
        self.t_contrastive = loading_dict["t_contrastive"]
        self.reconstruction_beta = loading_dict["reconstruction_beta"]
        self.contrastive_alpha = loading_dict["contrastive_alpha"]


        self.modalities = loading_dict["modalities"]

        self.mod_criteria = []
        for i in range(len_mod_criteria):
            # FIXME: On later versions we could try to not depend on pickle
            self.mod_criteria.append(
                torch.load(
                    osp.join(path,f"mod_criteria_{i}")
                )
            )


        self.model = SimpleMultimodalMultilossGraphAEModel(
            ds_in=ds_in,
            modalities_tgt_dist=self.mod_criteria,
            d_enc=self.d_enc,
        )
        self.model.load_state_dict(torch.load(osp.join(path,"model")))

        self.contrastive_modules = []
        for i in range(len_contrastive_modules):
            self.contrastive_modules.append(
                nn.Bilinear(self.d_enc[-1],self.d_enc[-1],1)
            )
            self.contrastive_modules[-1].load_state_dict(torch.load(osp.join(path,f"contrastive_modules_{i}")))

        self.opt = torch.optim.Adam(
            itertools.chain(
                self.model.parameters(),
                *[con_mod.parameters() for con_mod in self.contrastive_modules],
            ),
            lr=lr)
        self.opt.load_state_dict(torch.load(osp.join(path,"opt")))

        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt)
        self.sched.load_state_dict(torch.load(osp.join(path,"sched")))
        
        if osp.exists(osp.join(path,"history.json")):
            with open(osp.join(path,"history.json"), "r") as f:
                self.history = json.load(f)













