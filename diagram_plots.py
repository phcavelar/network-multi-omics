# %%
import os
import time
import warnings
from itertools import chain

import gdown

import numpy as np
import scipy as sp
import pandas as pd

import scanpy as sc

import seaborn as sns
import matplotlib.pyplot as plt

from nemo.models.subsetcontrastive import SubsetContrastive

import torch
import torch.nn as nn

import networkx as nx

# %%
DATA_PATH = "~/data/netemo"
DATA_PATH = os.path.expanduser(os.path.expandvars(DATA_PATH))
MODEL_DIR = "models"
RESULTS_PATH = "results"
SCRNA_FNAME = 'kotliarov2020-expressions.h5ad'
SCRNA_LINK = 'https://drive.google.com/uc?id=1wA3VBUnYEW2qHPk9WijNTKjV9KriWe8y',
SCCITE_FNAME = 'kotliarov2020-proteins.h5ad'
SCCITE_LINK = 'https://drive.google.com/uc?id=112mdDX76LZRL33tBLYhfYRRXOUrLUhw-'
NUM_REPS = 8

# %%
data = {}

for data_type, data_fname, data_link in [
            ("rna", SCRNA_FNAME, SCRNA_LINK),
            ("cite", SCCITE_FNAME, SCCITE_LINK),
        ]:
    try:
        data[data_type] = sc.read_h5ad(os.path.join(DATA_PATH,data_fname))
    except OSError:
        gdown.download(
            data_link,
            os.path.join(DATA_PATH,data_fname)
        )
        data[data_type] = sc.read_h5ad(os.path.join(DATA_PATH,data_fname))

# %%
#sc.pp.scale(data["rna"])
sc.tl.tsne(data["rna"])
sc.tl.tsne(data["cite"])

# %%
n_samples = data["rna"].X.shape[0]
sample_size = int(.1*n_samples)
min_overlap = 0.2

# %%
while True:
    idx_1:np.ndarray = np.random.choice(n_samples, size=sample_size, replace=False)
    # Guaranteed overlap
    idx_2_overlap = np.random.choice(idx_1, size=int(np.ceil(sample_size*min_overlap)), replace=False)
    choices_non_overlapping = np.array(sorted(set(np.arange(n_samples)).difference(set(set(idx_2_overlap)))))
    idx_2_others = np.random.choice(choices_non_overlapping, size=sample_size-int(np.ceil(sample_size*min_overlap)), replace=False)
    idx_2:np.ndarray = np.concatenate([idx_2_overlap,idx_2_others])
    np.random.shuffle(idx_2)
    del idx_2_overlap, choices_non_overlapping, idx_2_others
    #idx_2:np.ndarray = np.random.choice(n_samples, size=sample_size, replace=False)
    idx_1_s = set(idx_1)
    idx_2_s = set(idx_2)
    overlap = idx_1_s.intersection(idx_2_s)
    union = idx_1_s.union(idx_2_s)
    if len(overlap)/sample_size<min_overlap:
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
    for m in data:
        ad1 = sc.AnnData(data[m].X[idx_1])
        ad2 = sc.AnnData(data[m].X[idx_2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pp.neighbors(ad1)
            sc.pp.neighbors(ad2)
        As_1.append(np.stack(ad1.obsp["connectivities"].nonzero()))
        As_2.append(np.stack(ad2.obsp["connectivities"].nonzero()))
        ad1
        ad2
    break

# %%
g_full_rna = nx.Graph(zip(*data["rna"].obsp["connectivities"].nonzero()))
g_1_rna = nx.Graph(zip(*As_1[0]))
g_2_rna = nx.Graph(zip(*As_2[0]))

g_full_cite = nx.Graph(zip(*data["cite"].obsp["connectivities"].nonzero()))
g_full_cite.add_nodes_from(set(g_full_cite.nodes).symmetric_difference(np.arange(n_samples)))
g_1_cite = nx.Graph(zip(*As_1[1]))
g_2_cite = nx.Graph(zip(*As_2[1]))

# %%
colors = {
    "r": (*(np.array(sns.color_palette("colorblind")[3])*0.8),1.0),
    "g": (*(np.array(sns.color_palette("colorblind")[2])*0.8),1.0),
    "b": (*(np.array(sns.color_palette("colorblind")[0])*0.8),0.2),
    "y": (*(np.array(sns.color_palette("colorblind")[1])*0.8),1.0),
}

def get_c(i):
    if i in overlap:
        return colors["y"]
    elif i in idx_1_s:
        return colors["r"]
    elif i in idx_2_s:
        return colors["g"]
    return colors["b"]

# %%
nx.draw_networkx(g_full_rna, pos=data["rna"].obsm["X_tsne"], with_labels=False,
                 node_size=[5 if i in union else 1 for i in np.arange(n_samples)],
                 node_color=[get_c(i) for i in np.arange(n_samples)],
                 edge_color=(0.,0.,0.,0.1)
                 )
for s in plt.gca().spines.values():
    s.set_visible(False)
plt.savefig("graph_rna_full.png", transparent=True)
plt.close()

# %%
nx.draw_networkx(g_1_rna, pos=data["rna"].obsm["X_tsne"][idx_1,:], with_labels=False,
                 node_size=[3 if i in overlap else 1 for i in idx_1],
                 node_color=[get_c(i) for i in idx_1],
                 edge_color=(0.,0.,0.,0.1), width=1)
for s in plt.gca().spines.values():
    s.set_visible(False)
plt.savefig("graph_rna_sub1.png", transparent=True)
plt.close()

# %%
nx.draw_networkx(g_2_rna, pos=data["rna"].obsm["X_tsne"][idx_2,:], with_labels=False,
                 node_size=[3 if i in overlap else 1 for i in idx_2],
                 node_color=[get_c(i) for i in idx_2],
                 edge_color=(0.,0.,0.,0.1), width=1)
for s in plt.gca().spines.values():
    s.set_visible(False)
plt.savefig("graph_rna_sub2.png", transparent=True)
plt.close()

# %%
g_rna_overlap:nx.Graph = nx.relabel.convert_node_labels_to_integers(g_full_rna.subgraph(sorted(overlap)))
g_rna_overlap.clear_edges()
nx.draw_networkx(g_rna_overlap, pos=data["rna"].obsm["X_tsne"][sorted(overlap),:], with_labels=False,
                 node_size=[5 if i in union else 1 for i in sorted(overlap)],
                 node_color=[get_c(i) for i in sorted(overlap)],
                 edge_color=(0.,0.,0.,0.1)
                 )
for s in plt.gca().spines.values():
    s.set_visible(False)
plt.savefig("graph_rna_overlap.png", transparent=True)
plt.close()

# %%
nx.draw_networkx(g_full_cite, pos=data["cite"].obsm["X_tsne"], with_labels=False,
                 node_size=[5 if i in union else 1 for i in np.arange(n_samples)],
                 node_color=[get_c(i) for i in np.arange(n_samples)],
                 edge_color=(0.,0.,0.,0.1)
                 )
for s in plt.gca().spines.values():
    s.set_visible(False)
plt.savefig("graph_cite_full.png", transparent=True)
plt.close()

# %%
nx.draw_networkx(g_1_cite, pos=data["cite"].obsm["X_tsne"][idx_1,:], with_labels=False,
                 node_size=[3 if i in overlap else 1 for i in idx_1],
                 node_color=[get_c(i) for i in idx_1],
                 edge_color=(0.,0.,0.,0.1), width=1)
for s in plt.gca().spines.values():
    s.set_visible(False)
plt.savefig("graph_cite_sub1.png", transparent=True)
plt.close()

# %%
nx.draw_networkx(g_2_cite, pos=data["cite"].obsm["X_tsne"][idx_2,:], with_labels=False,
                 node_size=[3 if i in overlap else 1 for i in idx_2],
                 node_color=[get_c(i) for i in idx_2],
                 edge_color=(0.,0.,0.,0.1), width=1)
for s in plt.gca().spines.values():
    s.set_visible(False)
plt.savefig("graph_cite_sub2.png", transparent=True)
plt.close()

# %%
g_cite_overlap:nx.Graph = nx.relabel.convert_node_labels_to_integers(g_full_cite.subgraph(sorted(overlap)))
g_cite_overlap.clear_edges()
nx.draw_networkx(g_cite_overlap, pos=data["cite"].obsm["X_tsne"][sorted(overlap),:], with_labels=False,
                 node_size=[5 if i in union else 1 for i in sorted(overlap)],
                 node_color=[get_c(i) for i in sorted(overlap)],
                 edge_color=(0.,0.,0.,0.1)
                 )
for s in plt.gca().spines.values():
    s.set_visible(False)
plt.savefig("graph_cite_overlap.png", transparent=True)
plt.close()

# %%
use_avg = False
smol_obs_shape = 512 # 101.791015625 reduction
fsz_bigdim = max(plt.rcParams["figure.figsize"])
fsz_smldim = min(plt.rcParams["figure.figsize"])
sample_pct = sample_size/n_samples
cmap = "viridis"

#%%
celltype_key = "cluster_level3" #"cell_type"
prop_celltype = data["rna"].obs[celltype_key].value_counts()/data["rna"].obs[celltype_key].value_counts().sum()
total_n_celltype = data["rna"].obs[celltype_key].value_counts()
n_celltype = (np.ceil(prop_celltype*smol_obs_shape)).astype(int)
if n_celltype.sum()>smol_obs_shape:
    for i in range(n_celltype.sum()-smol_obs_shape):
        n_celltype.loc[np.random.choice(prop_celltype.index, p=prop_celltype.values)] -= 1
else:
    for i in range(smol_obs_shape-n_celltype.sum()):
        n_celltype.loc[np.random.choice(prop_celltype.index, p=prop_celltype.values)] += 1
n_celltype

proj_start_idx = 0
obs_proj = np.zeros((smol_obs_shape,n_samples))
proj_subset1 = []
proj_subset2 = []
for celltype in n_celltype.index:
    total_n = total_n_celltype[celltype]
    smol_n = n_celltype[celltype]
    num_celltype_choices = int(np.floor(total_n//smol_n))
    obs_choices = np.random.choice(total_n, (smol_n,num_celltype_choices), replace=False)
    # Subsets
    proj_subset_size = int(np.ceil(smol_n*sample_pct))
    this_ctype_range = np.arange(proj_start_idx,proj_start_idx+smol_n, dtype=int)
    proj1_ss = np.random.choice(this_ctype_range, proj_subset_size, replace=False)
    proj2_ss_ol = np.random.choice(proj1_ss, size=int(np.ceil(proj_subset_size*min_overlap)), replace=False)
    choices_non_overlapping = np.array(sorted(set(this_ctype_range).difference(set(set(proj2_ss_ol)))))
    proj2_ss_others = np.random.choice(choices_non_overlapping, size=proj_subset_size-int(np.ceil(proj_subset_size*min_overlap)), replace=False)
    proj2_ss:np.ndarray = np.concatenate([proj2_ss_ol,proj2_ss_others])
    np.random.shuffle(proj1_ss)
    proj_subset1.extend(proj1_ss)
    proj_subset2.extend(proj2_ss)
    # Projection
    for i, ichoice in enumerate(obs_choices, start=proj_start_idx):
        celltype_idxs = np.where(data["rna"].obs[celltype_key]==celltype)[0]
        obs_proj[i, celltype_idxs[ichoice]] = 1/len(ichoice) if use_avg else 1
    proj_start_idx += smol_n

proj_subset1 = [int(i) for i in proj_subset1]
proj_subset2 = [int(i) for i in proj_subset2]

# %%
rna_n_vars = data["rna"].X.shape[1]
rna_smol_var_shape = 400 # 9.9975 reduction
rna_switch_var_proj_method = "var"

match rna_switch_var_proj_method:
    case "corr":
        corrmat = np.corrcoef((obs_proj @ data["rna"].X).T)
        corrg = nx.Graph(np.abs(corrmat)>0.25)
        corrcom = nx.community.louvain_communities(corrg)
        var_choices_corr = [c for c in corrcom if len(c)>5]
        smol_var_shape_corr = len(var_choices_corr)
        smol_var_shape_rando = rna_smol_var_shape-smol_var_shape_corr
        num_var_choices = int(np.floor(rna_n_vars//smol_var_shape_rando))
        var_choices_rando = np.random.choice(rna_n_vars, (smol_var_shape_rando,num_var_choices), replace=False)
        rna_var_proj = np.zeros((rna_n_vars,rna_smol_var_shape))
        for i, ichoice in enumerate(var_choices_corr):
            rna_var_proj[list(ichoice),i] = 1/(len(ichoice)**0.25) if use_avg else 1
        for i, ichoice in enumerate(var_choices_rando, start=smol_var_shape_corr):
            rna_var_proj[ichoice,i] = 1 if use_avg else 1
    case "var": #Naive method was too empty
        rna_sd = np.std(data["rna"].X.A, axis=0)
        var_choices = np.argsort(rna_sd)[-rna_smol_var_shape:]
        rna_var_proj = np.zeros((rna_n_vars,rna_smol_var_shape))
        rna_var_proj[var_choices,np.arange(rna_smol_var_shape, dtype=int)] = 1
    case _: #Naive method was too empty
        num_var_choices = int(np.floor(rna_n_vars//rna_smol_var_shape))
        var_choices_rando = np.random.choice(rna_n_vars, (rna_smol_var_shape,num_var_choices), replace=False)
        rna_var_proj = np.zeros((rna_n_vars,rna_smol_var_shape))
        for i, ichoice in enumerate(var_choices_rando):
            rna_var_proj[ichoice,i] = 1/len(ichoice) if use_avg else 1

# %%
cite_n_vars = data["cite"].X.shape[1]
cite_smol_var_shape = 40 # 2.175 reduction
cite_switch_var_proj_method = "naive"

match cite_switch_var_proj_method:
    case "corr":
        corrmat = np.corrcoef((obs_proj @ data["cite"].X).T)
        corrg = nx.Graph(np.abs(corrmat)>0.25)
        corrcom = nx.community.louvain_communities(corrg)
        var_choices_corr = [c for c in corrcom if len(c)>5]
        smol_var_shape_corr = len(var_choices_corr)
        smol_var_shape_rando = cite_smol_var_shape-smol_var_shape_corr
        num_var_choices = int(np.floor(cite_n_vars//smol_var_shape_rando))
        var_choices_rando = np.random.choice(cite_n_vars, (smol_var_shape_rando,num_var_choices), replace=False)
        cite_var_proj = np.zeros((cite_n_vars,cite_smol_var_shape))
        for i, ichoice in enumerate(var_choices_corr):
            cite_var_proj[list(ichoice),i] = 1/(len(ichoice)**0.25) if use_avg else 1
        for i, ichoice in enumerate(var_choices_rando, start=smol_var_shape_corr):
            cite_var_proj[ichoice,i] = 1 if use_avg else 1
    case "var": #Naive method was too empty
        cite_sd = np.std(data["cite"].X, axis=0)
        var_choices = np.argsort(cite_sd)[-cite_smol_var_shape:]
        cite_var_proj = np.zeros((cite_n_vars,cite_smol_var_shape))
        cite_var_proj[var_choices,np.arange(cite_smol_var_shape, dtype=int)] = 1
    case _: #Naive method was too empty
        num_var_choices = int(np.floor(cite_n_vars//cite_smol_var_shape))
        var_choices_rando = np.random.choice(cite_n_vars, (cite_smol_var_shape,num_var_choices), replace=False)
        cite_var_proj = np.zeros((cite_n_vars,cite_smol_var_shape))
        for i, ichoice in enumerate(var_choices_rando):
            cite_var_proj[ichoice,i] = 1/len(ichoice) if use_avg else 1

#%%
def norm(X:np.ndarray) -> np.ndarray:
    return X/np.max(X, axis=0, keepdims=True)
    #return X/np.linalg.norm(X)

# %%
# rnafull
proj_shape = np.array([rna_var_proj.shape[1], obs_proj.shape[0]])
figsize = (fsz_bigdim*np.array(proj_shape)/np.max(proj_shape))
fig = plt.figure(figsize=figsize)
ax = plt.gca()
rna_proj_X = obs_proj @ data["rna"].X @ rna_var_proj
rna_proj_X = norm(rna_proj_X)
sns.heatmap(rna_proj_X, ax=ax, cbar=False, xticklabels=False, yticklabels=False)
sns.despine()
#aximg = plt.imshow()
for s in ax.spines.values():
    s.set_visible(False)
#plt.savefig("mat_rna_full.png")
plt.close()
plt.imsave("mat_rna_full_raw.png", rna_proj_X, cmap=cmap)

# %%
# rnasub1
proj_shape = np.array([rna_var_proj.shape[1], sample_pct*obs_proj.shape[0]])
figsize = (fsz_bigdim*np.array(proj_shape)/np.max(proj_shape))
fig = plt.figure(figsize=figsize)
ax = plt.gca()
proj_X = obs_proj[:,idx_1] @ data["rna"].X[idx_1,:] @ rna_var_proj#obs_proj @ data["rna"].X @ rna_var_proj#
proj_X = norm(proj_X)
#proj_X = proj_X[:int(np.ceil(proj_X.shape[0]*sample_pct)),:]
proj_X = proj_X[proj_subset1,:]
sns.heatmap(proj_X, ax=ax, cbar=False, xticklabels=False, yticklabels=False)
sns.despine()
#aximg = plt.imshow()
for s in ax.spines.values():
    s.set_visible(False)
#plt.savefig("mat_rna_sub1.png")
plt.close()
plt.imsave("mat_rna_sub1_raw.png", proj_X, cmap=cmap, vmin=np.min(proj_X), vmax=np.max(proj_X))

# %%
# rnasub2
proj_shape = np.array([rna_var_proj.shape[1], sample_pct*obs_proj.shape[0]])
figsize = (fsz_bigdim*np.array(proj_shape)/np.max(proj_shape))
fig = plt.figure(figsize=figsize)
ax = plt.gca()
proj_X = obs_proj[:,idx_2] @ data["rna"].X[idx_2,:] @ rna_var_proj#obs_proj @ data["rna"].X @ rna_var_proj#
proj_X = norm(proj_X)
proj_X = proj_X[proj_subset2,:]
sns.heatmap(proj_X, ax=ax, cbar=False, xticklabels=False, yticklabels=False)
sns.despine()
#aximg = plt.imshow()
for s in ax.spines.values():
    s.set_visible(False)
#plt.savefig("mat_rna_sub2.png")
plt.close()
plt.imsave("mat_rna_sub2_raw.png", proj_X, cmap=cmap, vmin=np.min(proj_X), vmax=np.max(proj_X))


# %%
# citefull
proj_shape = np.array([cite_var_proj.shape[1], obs_proj.shape[0]])
figsize = (fsz_bigdim*np.array(proj_shape)/np.max(proj_shape))
fig = plt.figure(figsize=figsize)
ax = plt.gca()
cite_proj_X = obs_proj @ data["cite"].X @ cite_var_proj
cite_proj_X = norm(cite_proj_X)
sns.heatmap(cite_proj_X, ax=ax, cbar=False, xticklabels=False, yticklabels=False)
sns.despine()
#aximg = plt.imshow()
for s in ax.spines.values():
    s.set_visible(False)
#plt.savefig("mat_cite_full.png")
plt.close()
plt.imsave("mat_cite_full_raw.png", cite_proj_X, cmap=cmap)

# %%
# citesub1
proj_shape = np.array([cite_var_proj.shape[1], sample_pct*obs_proj.shape[0]])
figsize = (fsz_bigdim*np.array(proj_shape)/np.max(proj_shape))
fig = plt.figure(figsize=figsize)
ax = plt.gca()
proj_X = obs_proj[:,idx_1] @ data["cite"].X[idx_1,:] @ cite_var_proj#obs_proj @ data["cite"].X @ cite_var_proj#
proj_X = norm(proj_X)
proj_X = proj_X[proj_subset1,:]
sns.heatmap(proj_X, ax=ax, cbar=False, xticklabels=False, yticklabels=False)
sns.despine()
#aximg = plt.imshow()
for s in ax.spines.values():
    s.set_visible(False)
#plt.savefig("mat_cite_sub1.png")
plt.close()
plt.imsave("mat_cite_sub1_raw.png", proj_X, cmap=cmap, vmin=np.min(cite_proj_X), vmax=np.max(cite_proj_X))

# %%
# citesub2
proj_shape = np.array([cite_var_proj.shape[1], sample_pct*obs_proj.shape[0]])
figsize = (fsz_bigdim*np.array(proj_shape)/np.max(proj_shape))
fig = plt.figure(figsize=figsize)
ax = plt.gca()
proj_X = obs_proj[:,idx_2] @ data["cite"].X[idx_2,:] @ cite_var_proj#obs_proj @ data["cite"].X @ cite_var_proj#
proj_X = norm(proj_X)
proj_X = proj_X[proj_subset2,:]
sns.heatmap(proj_X, ax=ax, cbar=False, xticklabels=False, yticklabels=False)
sns.despine()
#aximg = plt.imshow()
for s in ax.spines.values():
    s.set_visible(False)
#plt.savefig("mat_cite_sub2.png")
plt.close()
plt.imsave("mat_cite_sub2_raw.png", proj_X, cmap=cmap, vmin=np.min(cite_proj_X), vmax=np.max(cite_proj_X))

#%%
rec_error = 0.01

# %%
# rnasub1_rec
proj_shape = np.array([rna_var_proj.shape[1], sample_pct*obs_proj.shape[0]])
figsize = (fsz_bigdim*np.array(proj_shape)/np.max(proj_shape))
fig = plt.figure(figsize=figsize)
ax = plt.gca()
proj_X = obs_proj[:,idx_1] @ data["rna"].X[idx_1,:] @ rna_var_proj#obs_proj @ data["rna"].X @ rna_var_proj#
proj_X += np.random.normal(0,np.linalg.norm(proj_X, axis=0)*rec_error, proj_X.shape)
proj_X = norm(proj_X)
#proj_X = proj_X[:int(np.ceil(proj_X.shape[0]*sample_pct)),:]
proj_X = proj_X[proj_subset1,:]
sns.heatmap(proj_X, ax=ax, cbar=False, xticklabels=False, yticklabels=False)
sns.despine()
#aximg = plt.imshow()
for s in ax.spines.values():
    s.set_visible(False)
#plt.savefig("mat_rna_sub1_rec.png")
plt.close()
plt.imsave("mat_rna_sub1_rec_raw.png", proj_X, cmap=cmap, vmin=np.min(proj_X), vmax=np.max(proj_X))

# %%
# rnasub2_rec
proj_shape = np.array([rna_var_proj.shape[1], sample_pct*obs_proj.shape[0]])
figsize = (fsz_bigdim*np.array(proj_shape)/np.max(proj_shape))
fig = plt.figure(figsize=figsize)
ax = plt.gca()
proj_X = obs_proj[:,idx_2] @ data["rna"].X[idx_2,:] @ rna_var_proj#obs_proj @ data["rna"].X @ rna_var_proj#
proj_X += np.random.normal(0,np.linalg.norm(proj_X, axis=0)*rec_error, proj_X.shape)
proj_X = norm(proj_X)
proj_X = proj_X[proj_subset2,:]
sns.heatmap(proj_X, ax=ax, cbar=False, xticklabels=False, yticklabels=False)
sns.despine()
#aximg = plt.imshow()
for s in ax.spines.values():
    s.set_visible(False)
#plt.savefig("mat_rna_sub2_rec.png")
plt.close()
plt.imsave("mat_rna_sub2_rec_raw.png", proj_X, cmap=cmap, vmin=np.min(proj_X), vmax=np.max(proj_X))

#%%
rec_error = 0.05

# %%
# citesub1_rec
proj_shape = np.array([cite_var_proj.shape[1], sample_pct*obs_proj.shape[0]])
figsize = (fsz_bigdim*np.array(proj_shape)/np.max(proj_shape))
fig = plt.figure(figsize=figsize)
ax = plt.gca()
proj_X = obs_proj[:,idx_1] @ data["cite"].X[idx_1,:] @ cite_var_proj#obs_proj @ data["cite"].X @ cite_var_proj#
proj_X += np.random.normal(0,np.linalg.norm(proj_X, axis=0)*rec_error, proj_X.shape)
proj_X = norm(proj_X)
proj_X = proj_X[proj_subset1,:]
sns.heatmap(proj_X, ax=ax, cbar=False, xticklabels=False, yticklabels=False)
sns.despine()
#aximg = plt.imshow()
for s in ax.spines.values():
    s.set_visible(False)
#plt.savefig("mat_cite_sub1_rec.png")
plt.close()
plt.imsave("mat_cite_sub1_rec_raw.png", proj_X, cmap=cmap, vmin=np.min(cite_proj_X), vmax=np.max(cite_proj_X))

# %%
# citesub2_rec
proj_shape = np.array([cite_var_proj.shape[1], sample_pct*obs_proj.shape[0]])
figsize = (fsz_bigdim*np.array(proj_shape)/np.max(proj_shape))
fig = plt.figure(figsize=figsize)
ax = plt.gca()
proj_X = obs_proj[:,idx_2] @ data["cite"].X[idx_2,:] @ cite_var_proj#obs_proj @ data["cite"].X @ cite_var_proj#
proj_X += np.random.normal(0,np.linalg.norm(proj_X, axis=0)*rec_error, proj_X.shape)
proj_X = norm(proj_X)
proj_X = proj_X[proj_subset2,:]
sns.heatmap(proj_X, ax=ax, cbar=False, xticklabels=False, yticklabels=False)
sns.despine()
#aximg = plt.imshow()
for s in ax.spines.values():
    s.set_visible(False)
#plt.savefig("mat_cite_sub2_rec.png")
plt.close()
plt.imsave("mat_cite_sub2_rec_raw.png", proj_X, cmap=cmap, vmin=np.min(cite_proj_X), vmax=np.max(cite_proj_X))

# %%


# %%


