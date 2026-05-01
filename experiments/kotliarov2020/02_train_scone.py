# %%
import os
from itertools import chain

import gdown

import numpy as np
import pandas as pd

import scanpy as sc

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics.cluster import silhouette_samples

from nemo.models.subsetcontrastive import SubsetContrastive

import torch
import torch.nn as nn

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
sc.pp.scale(data["rna"])

# %%
num_plots=10
plot_df = pd.melt(pd.DataFrame(data["cite"].layers["count"][:,:num_plots].A, columns=data["cite"].var.index[:num_plots], index=data["cite"].obs.index).reset_index(), id_vars="index")
g = sns.kdeplot(data=plot_df, x="value", hue="variable", common_norm=False, legend=False)
plt.show()

# %%
num_plots=10
plot_df = pd.melt(pd.DataFrame(data["cite"].X[:,:num_plots], columns=data["cite"].var.index[:num_plots], index=data["cite"].obs.index).reset_index(), id_vars="index")
g = sns.kdeplot(data=plot_df, x="value", hue="variable", common_norm=False, legend=False)
plt.show()

# %%
#torch.set_float32_matmul_precision('medium') #Did not provide any significant speedup

for repetition in range(NUM_REPS):
    gae = SubsetContrastive(
        data,
        {"rna":nn.MSELoss(),"cite":nn.MSELoss()},
        0.2,
    )
    tr_log = gae.train(data)
    Xs = [torch.tensor(data[m].X) for m in gae.modalities]
    As = [torch.tensor(np.stack(data[m].obsp["connectivities"].nonzero()), dtype=torch.long) for m in gae.modalities]
    _, __, z = gae.model(Xs, As)

    adz = sc.AnnData(z.detach().cpu().numpy())
    adz.obs = data["rna"].obs

    os.makedirs(os.path.join(DATA_PATH,RESULTS_PATH), exist_ok=True)
    adz.write(os.path.join(DATA_PATH,RESULTS_PATH,f"kotliarov2020-subsetcontrastive-{repetition}.h5ad"))
    os.makedirs(os.path.join(DATA_PATH,MODEL_DIR), exist_ok=True)
    gae.save(os.path.join(DATA_PATH,MODEL_DIR,f"kotliarov2020-subsetcontrastive-{repetition}.model"))
    history_df = pd.DataFrame(tr_log)
    history_df.to_csv(os.path.join(DATA_PATH,RESULTS_PATH,f"kotliarov2020-subsetcontrastive-{repetition}-history.csv.gz"))

# %%
# Load first model for diagnostic plots
z = sc.read_h5ad(os.path.join(DATA_PATH,RESULTS_PATH,"kotliarov2020-subsetcontrastive-0.h5ad"))
history_df = pd.read_csv(os.path.join(DATA_PATH,RESULTS_PATH,"kotliarov2020-subsetcontrastive-0-history.csv.gz"))

# %%
sc.pp.neighbors(z)
sc.tl.pca(z)
sc.pl.pca(z, color=['batch', 'cell_type', 'cluster_level2', 'cluster_level3'], ncols=1)
# %%
sc.pp.neighbors(z)
sc.tl.umap(z)
sc.pl.umap(z, color=['batch', 'cell_type', 'cluster_level2', 'cluster_level3'], ncols=1)
# %%
sc.tl.tsne(z)
sc.pl.tsne(z, color=['batch', 'cell_type', 'cluster_level2', 'cluster_level3'], ncols=1)
#%%
sc.pl.umap(z, color=['batch', 'cell_type',], ncols=2)
# %%
sc.pp.neighbors(z, use_rep="X")
sc.tl.pca(z)
sc.pl.pca(z, color=['batch', 'cell_type', 'cluster_level2', 'cluster_level3'], ncols=1)
# %%
sc.pp.neighbors(z, use_rep="X")
sc.tl.umap(z)
sc.pl.umap(z, color=['batch', 'cell_type', 'cluster_level2', 'cluster_level3'], ncols=1)
#%%
sc.pl.umap(z, color=['batch', 'cell_type',], ncols=2)
#%%
sc.pl.umap(z, color=['cluster_level2', 'cluster_level3'], ncols=1)
#%%
sc.pl.umap(z, color=['cluster_level2'], ncols=1)
#%%
sc.pl.umap(z, color=['cluster_level3'], ncols=1)

# %%
axes = sc.pl.tsne(z, color=['batch', 'cell_type', 'cluster_level3'], ncols=2, show=False)
axes[0].set_title("Batch")
axes[1].set_title("Cell Type (broad)")
axes[2].set_title("Cell Type (fine)")
plt.savefig("tsne-kotliarov2020-subsetcontrastive-celltypes.pdf")
plt.savefig("tsne-kotliarov2020-subsetcontrastive-celltypes-200.png", dpi=200)
plt.savefig("tsne-kotliarov2020-subsetcontrastive-celltypes-300.png", dpi=300)
plt.savefig("tsne-kotliarov2020-subsetcontrastive-celltypes-600.png", dpi=600)

#%%
cell_proportions_broad = 100.*z.obs["cell_type"].value_counts()/z.obs.shape[0]
cell_proportions_fine = 100.*z.obs["cluster_level3"].value_counts()/z.obs.shape[0]

#%%
z.obs[f"ss_l1_cos"] = silhouette_samples(z.X, z.obs[f"cell_type"], metric="cosine")
z.obs[f"ss_l1_euc"] = silhouette_samples(z.X, z.obs[f"cell_type"], metric="euclidean")
z.obs[f"ss_l2_cos"] = silhouette_samples(z.X, z.obs[f"cluster_level3"], metric="cosine")
z.obs[f"ss_l2_euc"] = silhouette_samples(z.X, z.obs[f"cluster_level3"], metric="euclidean")

#%%
broad_prop_ss = pd.merge(
    cell_proportions_broad,
    z.obs.groupby("cell_type")[["ss_l1_cos","ss_l1_euc"]].mean(), left_index=True, right_index=True
)
broad_prop_ss
#%%
print("Broad SS: ", (broad_prop_ss["count"]*broad_prop_ss["ss_l1_cos"]).sum()/broad_prop_ss["count"].sum())
print("Fine common SS: ", (broad_prop_ss[broad_prop_ss["count"]>=1.]["count"]*broad_prop_ss[broad_prop_ss["count"]>=1.]["ss_l1_cos"]).sum()/broad_prop_ss[broad_prop_ss["count"]>=1.]["count"].sum())
print("Fine rare SS: ", (broad_prop_ss[broad_prop_ss["count"]<1.]["count"]*broad_prop_ss[broad_prop_ss["count"]<1.]["ss_l1_cos"]).sum()/broad_prop_ss[broad_prop_ss["count"]<1.]["count"].sum())
print(broad_prop_ss[["count","ss_l1_cos"]].to_latex())
#%%
fine_prop_ss = pd.merge(
    cell_proportions_fine,
    z.obs.groupby("cluster_level3")[["ss_l2_cos","ss_l2_euc"]].mean(), left_index=True, right_index=True
)
fine_prop_ss
#%%
print("Fine SS: ", (broad_prop_ss["count"]*broad_prop_ss["ss_l1_cos"]).sum()/broad_prop_ss["count"].sum())
print("Fine common SS: ", (fine_prop_ss[fine_prop_ss["count"]>=1.]["count"]*fine_prop_ss[fine_prop_ss["count"]>=1.]["ss_l2_cos"]).sum()/fine_prop_ss[fine_prop_ss["count"]>=1.]["count"].sum())
print("Fine rare SS: ", (fine_prop_ss[fine_prop_ss["count"]<1.]["count"]*fine_prop_ss[fine_prop_ss["count"]<1.]["ss_l2_cos"]).sum()/fine_prop_ss[fine_prop_ss["count"]<1.]["count"].sum())
print(fine_prop_ss[["count","ss_l2_cos"]].to_latex())

#%%
paper_markers = {
    "T-cell": [
        f"{p}_PROT" for p in ["CD2", "CD3", "CD5", "CD7", "CD4", "CD8", "CD62L", "CD45RA", "CD45RO", "CD27", "CD28", "CD278 ", "CD25", "CD127", "CD161", "KLRG1", "CD195", "CD314 ", "CD194", "CD103",]
    ],
    "NK": [
        f"{p}_PROT" for p in ["CD56", "CD57", "CD244", "CD16",]
    ],
    "Monocytes": [
        f"{p}_PROT" for p in ["CD244", "CD16", "CD14", "CD11b", "CD11c", "CD1d", "CD33", "CD13", "CD31", "CD64", "CD163", "CD86", "HLA-DR", "CD123", "CD141", "CD71",]
    ],
    "pDC": [
        f"{p}_PROT" for p in ["HLA-DR","CD123","CD141","CD71","CD303",]
    ],
    "HSC": [
        f"{p}_PROT" for p in ["CD117", "CD34",]
    ],
    "B": [
        f"{p}_PROT" for p in ["CD38", "CD39", "CD1c", "CD32", "IgM", "IgD", "IgA", "CD19", "CD20", "CD21", "CD24", "CD40", "CD185", "CD196", "BTLA",]
    ],
}

paper_markers_i = {
    k: [data["cite"].var_names.get_loc(vi) for vi in paper_markers[k]] for k in paper_markers
}

full_paper_markers = list(chain(*[paper_markers[k] for k in paper_markers]))

celltype_to_paper_markers = {
    'NK': ["NK"],
    'CD8 naive': ["T-cell"],
    'CD4 naive': ["T-cell"],
    'Classical monocytes and mDC': ["Monocytes", "HSC"],
    'CD8 memory T': ["T-cell"],
    'CD4 memory T': ["T-cell"],
    'B': ["B"],
    'Unconventional T cells': ["T-cell"],
    'Non-classical monocytes': ["Monocytes"],
    'pDC': ["pDC"],
}

paper_celltype_markers = {
    'NK': list(chain(paper_markers["NK"],)),
    'CD8 naive': list(chain(paper_markers["T-cell"],)),
    'CD4 naive': list(chain(paper_markers["T-cell"],)),
    'Classical monocytes and mDC': list(chain(paper_markers["Monocytes"],paper_markers["HSC"],)),
    'CD8 memory T': list(chain(paper_markers["T-cell"],)),
    'CD4 memory T': list(chain(paper_markers["T-cell"],)),
    'B': list(chain(paper_markers["B"],)),
    'Unconventional T cells': list(chain(paper_markers["T-cell"],)),
    'Non-classical monocytes': list(chain(paper_markers["Monocytes"],)),
    'pDC': list(chain(paper_markers["pDC"],))
}

paper_celltype_markers_i = {
    k: [data["cite"].var_names.get_loc(vi) for vi in paper_celltype_markers[k]] for k in paper_celltype_markers
}

for k in paper_celltype_markers_i:
    z.obs[f"Marker Celltype = {k}"] = pd.Series(np.mean(data["cite"].X[:,paper_celltype_markers_i[k]], axis=1), data["cite"].obs_names)
for k in paper_markers_i:
    z.obs[f"Marker {k}"] = pd.Series(np.mean(data["cite"].X[:,paper_markers_i[k]], axis=1), data["cite"].obs_names)

#%%
for k in paper_celltype_markers_i:
    sc.pl.umap(z, color=[f"Marker Celltype = {k}", "cell_type",], ncols=2)

#%%
for k in paper_celltype_markers_i:
    sc.pl.pca(z, color=[f"Marker Celltype = {k}", "cell_type",], ncols=2)
#%%
for k in paper_celltype_markers_i:
    sc.pl.tsne(z, color=[f"Marker Celltype = {k}", "cell_type",], ncols=2)
#%%
for k in paper_markers_i:
    sc.pl.tsne(z, color=[f"Marker {k}", "cell_type",], ncols=2)

#%%
axes = sc.pl.tsne(z, color=[*[f"Marker {k}" for k in paper_markers_i]], ncols=3, show=False)
for i, k in enumerate([*[f"Marker {k}" for k in paper_markers_i]]):
    match k:
        case "B":
            k = "B-cell"
        case "cell_type":
            k = "Cell Type"
        case _:
            pass
    axes[i].set_title(k)
plt.savefig("tsne-kotliarov2020-subsetcontrastive-markers.pdf")
plt.savefig("tsne-kotliarov2020-subsetcontrastive-markers-200.png", dpi=200)
plt.savefig("tsne-kotliarov2020-subsetcontrastive-markers-300.png", dpi=300)
plt.savefig("tsne-kotliarov2020-subsetcontrastive-markers-600.png", dpi=600)

#%%
for k in paper_markers_i:
    for p, i in zip(paper_markers[k], paper_markers_i[k]):
        p = p.split("_PROT")[0]
        z.obs[f"{k} marker: {p}"] = pd.Series(data["cite"].X[:,i], data["cite"].obs_names)
    axes = sc.pl.tsne(z, color=[*[f"{k} marker: {p.split('_PROT')[0]}" for p in paper_markers[k]]], ncols=3, show=False)
    plt.savefig(f"tsne-kotliarov2020-subsetcontrastive-markers-{k}.pdf")
    plt.savefig(f"tsne-kotliarov2020-subsetcontrastive-markers-{k}-200.png", dpi=200)
    plt.close()

#%%
#sc.pl.heatmap(z, color=[f"Marker_{k}", "cell_type",], ncols=2)

# %%

history_plot_keys = [
    ['total', 'r', 'c'],
    ['r', 'r_0', 'r_1',],
    ['c', 'c1', 'c2',],
    ['c1', 'c1_0', 'c1_1',],
    ['c2', 'c2_0', 'c2_1',],
    ['|∇|',],
]

for this_same_plot_keys in history_plot_keys:
    plot_df = history_df[this_same_plot_keys]

    q_val = np.quantile(plot_df.values, 0.98)
    ymin = np.min(plot_df.values)
    ymax = np.max(plot_df.values)

    if ymax>1.1*q_val:
        fig, (axes) = plt.subplots(1, 2, figsize=(16,6))
        sns.lineplot(plot_df, ax=axes[0])
        sns.lineplot(plot_df, ax=axes[1])
        axymin = axes[1].get_ylim()[0]
        
        axes[1].set_ylim(np.average([axymin,ymin], weights=[1,3]), q_val)
    else:
        fig, (ax) = plt.subplots(1, 1, figsize=(8,6))
        sns.lineplot(plot_df, ax=ax)

    plt.show()

# %%



