# %%
import os

import gdown

import numpy as np
import pandas as pd

import scanpy as sc

import seaborn as sns
import matplotlib.pyplot as plt

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
            #("cite", SCCITE_FNAME, SCCITE_LINK),
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
        {"rna":nn.MSELoss()},#,"cite":nn.MSELoss()},
        0.2,
    )
    tr_log = gae.train(data)
    Xs = [torch.tensor(data[m].X) for m in gae.modalities]
    As = [torch.tensor(np.stack(data[m].obsp["connectivities"].nonzero()), dtype=torch.long) for m in gae.modalities]
    _, __, z = gae.model(Xs, As)

    adz = sc.AnnData(z.detach().cpu().numpy())
    adz.obs = data["rna"].obs

    os.makedirs(os.path.join(DATA_PATH,RESULTS_PATH), exist_ok=True)
    adz.write(os.path.join(DATA_PATH,RESULTS_PATH,f"kotliarov2020-subsetcontrastiveRNA-{repetition}.h5ad"))
    os.makedirs(os.path.join(DATA_PATH,MODEL_DIR), exist_ok=True)
    gae.save(os.path.join(DATA_PATH,MODEL_DIR,f"kotliarov2020-subsetcontrastiveRNA-{repetition}.model"))
    history_df = pd.DataFrame(tr_log)
    history_df.to_csv(os.path.join(DATA_PATH,RESULTS_PATH,f"kotliarov2020-subsetcontrastiveRNA-{repetition}-history.csv.gz"))

# %%
# Load first model for diagnostic plots
z = sc.read_h5ad(os.path.join(DATA_PATH,RESULTS_PATH,"kotliarov2020-subsetcontrastiveRNA-0.h5ad"))
history_df = pd.read_csv(os.path.join(DATA_PATH,RESULTS_PATH,"kotliarov2020-subsetcontrastiveRNA-0-history.csv.gz"))

# %%
sc.pp.neighbors(z, use_rep='X')
sc.tl.umap(z)
sc.pl.umap(z, color=['batch', 'cell_type', 'cluster_level2', 'cluster_level3'], ncols=1)
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



