# %%
import os
import time
import gc
from itertools import chain

import gdown
from tqdm.autonotebook import tqdm

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as sps
import networkx as nx

import sklearn

import scanpy as sc

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics.cluster import silhouette_samples

from nemo.models.subsetcontrastive import SubsetContrastive
from nemo.graph import get_leiden, get_louvain

import torch
import torch.nn as nn

# %%
DATA_PATH = "~/data/netemo"
DATA_PATH = os.path.expanduser(os.path.expandvars(DATA_PATH))
MODEL_DIR = "models"
RESULTS_PATH = "results-ablation"
SCRNA_FNAME = 'kotliarov2020-expressions.h5ad'
SCRNA_LINK = 'https://drive.google.com/uc?id=1wA3VBUnYEW2qHPk9WijNTKjV9KriWe8y',
SCCITE_FNAME = 'kotliarov2020-proteins.h5ad'
SCCITE_LINK = 'https://drive.google.com/uc?id=112mdDX76LZRL33tBLYhfYRRXOUrLUhw-'
NUM_REPS = 4

#%%
KNN_KS = [15,5,10,20,30]
KNN_AS = {k:{} for k in KNN_KS}

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
    for k in KNN_KS:        
        KNN_AS[k][data_type] = torch.tensor(np.stack(sc.pp.neighbors(data[data_type], n_neighbors=k, copy=True).obsp["connectivities"].nonzero()), dtype=torch.long)

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
    for n_neighbors in [15]:
        for reconstruction_beta in [10,1,0] if n_neighbors==15 else [10]:
            for contrastive_alpha in [1,0] if n_neighbors==15 else [1]:
                anndata_fpath = os.path.join(DATA_PATH,RESULTS_PATH,f"kotliarov2020-subsetcontrastive-ablation-{repetition}-{n_neighbors}-{contrastive_alpha:.2f}-{reconstruction_beta:.2f}.h5ad")
                history_fpath = os.path.join(DATA_PATH,RESULTS_PATH,f"kotliarov2020-subsetcontrastive-ablation-{repetition}-{n_neighbors}-{contrastive_alpha:.2f}-{reconstruction_beta:.2f}-history.csv.gz")

                if (
                    (contrastive_alpha==0 and reconstruction_beta==0 )
                    or all(map(os.path.exists,[anndata_fpath,history_fpath]))
                    ):
                    continue

                print(repetition, n_neighbors, contrastive_alpha, reconstruction_beta)
                gae = SubsetContrastive(
                    data,
                    {"rna":nn.MSELoss(),"cite":nn.MSELoss()},
                    0.2,
                    reconstruction_beta=reconstruction_beta,
                    contrastive_alpha=contrastive_alpha,
                    n_neighbors=n_neighbors,
                )
                print("A", flush=True)
                tr_log = gae.train(data)
                print("B", flush=True)
                Xs = [torch.tensor(data[m].X) for m in gae.modalities]
                print("C", flush=True)
                #As = [torch.tensor(np.stack(data[m].obsp["connectivities"].nonzero()), dtype=torch.long) for m in gae.modalities]
                As = [KNN_AS[n_neighbors][m] for m in gae.modalities]
                print("D", flush=True)
                _, __, z = gae.model(Xs, As)
                print("E", flush=True)

                adz = sc.AnnData(z.detach().cpu().numpy())
                print("F", flush=True)
                adz.obs = data["rna"].obs
                print("G", flush=True)

                os.makedirs(os.path.join(DATA_PATH,RESULTS_PATH), exist_ok=True)
                adz.write(anndata_fpath)
                if repetition==0:
                    os.makedirs(os.path.join(DATA_PATH,MODEL_DIR), exist_ok=True)
                    gae.save(os.path.join(DATA_PATH,MODEL_DIR,f"kotliarov2020-subsetcontrastive-ablation-{repetition}-{n_neighbors}-{contrastive_alpha:.2f}-{reconstruction_beta:.2f}.model"))
                history_df = pd.DataFrame(tr_log)
                history_df.to_csv(history_fpath)

                del adz
                del gae

#%%
for n_neighbors in KNN_KS:
    if n_neighbors==15:
        continue
    for repetition in range(NUM_REPS):
        for reconstruction_beta in [10]:
            for contrastive_alpha in [1]:
                anndata_fpath = os.path.join(DATA_PATH,RESULTS_PATH,f"kotliarov2020-subsetcontrastive-ablation-{repetition}-{n_neighbors}-{contrastive_alpha:.2f}-{reconstruction_beta:.2f}.h5ad")
                history_fpath = os.path.join(DATA_PATH,RESULTS_PATH,f"kotliarov2020-subsetcontrastive-ablation-{repetition}-{n_neighbors}-{contrastive_alpha:.2f}-{reconstruction_beta:.2f}-history.csv.gz")

                if (
                    (contrastive_alpha==0 and reconstruction_beta==0 )
                    or all(map(os.path.exists,[anndata_fpath,history_fpath]))
                    ):
                    continue

                print(repetition, n_neighbors, contrastive_alpha, reconstruction_beta)
                gae = SubsetContrastive(
                    data,
                    {"rna":nn.MSELoss(),"cite":nn.MSELoss()},
                    0.2,
                    reconstruction_beta=reconstruction_beta,
                    contrastive_alpha=contrastive_alpha,
                    n_neighbors=n_neighbors,
                )
                print("A", flush=True)
                tr_log = gae.train(data)
                print("B", flush=True)
                Xs = [torch.tensor(data[m].X) for m in gae.modalities]
                print("C", flush=True)
                #As = [torch.tensor(np.stack(data[m].obsp["connectivities"].nonzero()), dtype=torch.long) for m in gae.modalities]
                As = [KNN_AS[n_neighbors][m] for m in gae.modalities]
                print("D", flush=True)
                _, __, z = gae.model(Xs, As)
                print("E", flush=True)

                adz = sc.AnnData(z.detach().cpu().numpy())
                print("F", flush=True)
                adz.obs = data["rna"].obs
                print("G", flush=True)

                os.makedirs(os.path.join(DATA_PATH,RESULTS_PATH), exist_ok=True)
                adz.write(anndata_fpath)
                if repetition==0:
                    os.makedirs(os.path.join(DATA_PATH,MODEL_DIR), exist_ok=True)
                    gae.save(os.path.join(DATA_PATH,MODEL_DIR,f"kotliarov2020-subsetcontrastive-ablation-{repetition}-{n_neighbors}-{contrastive_alpha:.2f}-{reconstruction_beta:.2f}.model"))
                history_df = pd.DataFrame(tr_log)
                history_df.to_csv(history_fpath)

                del adz
                del gae

# %%
# Load first model for diagnostic plots
del data
gc.collect()

#%%
models = [
    #"totalvi", "totalviRNA", "totalviPRO",
    "subsetcontrastive", #"subsetcontrastiveRNA", "subsetcontrastivePRO",
    #"MANE",
]
models_with_subfolders = {
    #"MANE": lambda basefolder: os.path.join(basefolder, "cmp", "mane", "kotliarov2020", "emb_{rep}.npy"),
}

resolution_resolution = 20
resolutions = np.linspace(0,2,resolution_resolution+1,endpoint=True)[1:]
resolutions

# %%
try:
    results_df = pd.read_csv("kotliarov2020-ablation-results.csv")
    results_dict = results_df.to_dict("list")
    if "Unnamed: 0" in results_dict: del results_dict["Unnamed: 0"]
except FileNotFoundError:
    results_dict = {
        "model": [],
        "rep": [],
        "n_neighbors": [],
        "reconstruction_beta": [],
        "contrastive_alpha": [],
        #"louvain_r": [],
        #"louvain_mod_nx": [],
        #"louvain_ami": [],
        #"louvain_ari": [],
        "leiden_r": [],
        "leiden_mod_nx": [],
        "leiden_mod_ig": [],
        "leiden_ami": [],
        "leiden_ari": [],
    }

# %%
ignored_errors = []
interrupted = False
existing_model_rep_pairs = set(zip(results_dict["model"], results_dict["rep"]))
with tqdm(total=len(models),) as counter:
    for model in models:
        for rep in range(NUM_REPS):
            for n_neighbors in KNN_KS:
                betas_options = [10,1,0] if n_neighbors==15 else [10]
                alphas_options = [1,0] if n_neighbors==15 else [1]
                for reconstruction_beta in betas_options:
                    for contrastive_alpha in alphas_options:
                        try:
                            if (model, rep) in existing_model_rep_pairs:
                                counter.write(f"{model} {rep} already calculated before, skipping")
                                continue
                            counter.set_description(f"{model} {rep}/{NUM_REPS}")
                            if model not in models_with_subfolders:
                                anndata = sc.read_h5ad(os.path.join(DATA_PATH,RESULTS_PATH,f"kotliarov2020-{model}-ablation-{rep}-{n_neighbors}-{contrastive_alpha:.2f}-{reconstruction_beta:.2f}.h5ad"))
                                if "cell_type" not in anndata.obs:
                                    anndata.obs = sc.read_h5ad(os.path.join(DATA_PATH,SCRNA_FNAME)).obs
                                #history_df = pd.read_csv(os.path.join(DATA_PATH,RESULTS_PATH,f"kotliarov2020-{model}-{rep}-history.csv.gz"))
                            else:
                                fpath = models_with_subfolders[model](os.path.join(DATA_PATH,RESULTS_PATH))
                                fpath = fpath.format(rep=rep)
                                counter.write(fpath)
                                try:
                                    if fpath.endswith(".npy"):
                                        print(".npy")
                                        anndata = sc.AnnData(np.load(fpath))
                                    elif fpath.endswith(".csv"):
                                        print(".csv")
                                        df = pd.read_csv(fpath, index_col=0)
                                        anndata = sc.AnnData(df.X)
                                    elif fpath.endswith(".h5ad"):
                                        print(".h5ad")
                                        anndata = sc.read_h5ad(fpath)
                                    else:
                                        raise ValueError(fpath)
                                except FileNotFoundError as e:
                                    ignored_errors.append(e)
                                    counter.update(1/NUM_REPS)
                                    continue
                                if "cell_type" not in anndata.obs:
                                    anndata.obs = sc.read_h5ad(os.path.join(DATA_PATH,SCRNA_FNAME)).obs

                            sc.pp.neighbors(anndata, n_neighbors=n_neighbors, use_rep='X')
                            g = nx.Graph(anndata.obsp["connectivities"])


                            if False:
                                counter.set_description(f"{model} {rep}/{NUM_REPS} louvain")
                                clus_louvain_dict = {
                                    res: get_louvain(g, resolution=res)
                                    for res in resolutions
                                }

                                results_louvain = sorted(
                                    [
                                        (r,
                                        nx.community.modularity(g,clus_louvain_dict[r][0]),
                                        sklearn.metrics.adjusted_mutual_info_score(anndata.obs["cell_type"], clus_louvain_dict[r][1]),
                                        sklearn.metrics.adjusted_rand_score(anndata.obs["cell_type"], clus_louvain_dict[r][1]),
                                        ) for r in resolutions
                                    ],
                                    key=lambda x:x[1],
                                    reverse=True
                                )

                            counter.set_description(f"{model} {rep}/{NUM_REPS} leiden")
                            clus_leiden_dict = {
                                res: get_leiden(anndata.obsp["connectivities"], resolution=res)
                                for res in resolutions
                            }
                            results_leiden = sorted(
                                [
                                    (r,
                                    clus_leiden_dict[r].modularity,
                                    nx.community.modularity(g,[set((i for i,c in enumerate(clus_leiden_dict[r].membership) if c==k)) for k in sorted(set(clus_leiden_dict[r].membership))]),
                                    sklearn.metrics.adjusted_mutual_info_score(anndata.obs["cell_type"], clus_leiden_dict[r].membership),
                                    sklearn.metrics.adjusted_rand_score(anndata.obs["cell_type"], clus_leiden_dict[r].membership),
                                    ) for r in resolutions
                                ],
                                key=lambda x:x[1],
                                reverse=True
                            )
                            # Check that Leiden's internal modularity calculation give an equal ranking to networkx's
                            rla = np.array(results_leiden)
                            counter.write(f"{np.mean((rla[:,1:2]>rla[:,1:2].T)==(rla[:,2:3]>rla[:,2:3].T)):.2%} of ranking match")
                            counter.set_description(f"{model} {rep}/{NUM_REPS} wrapup")
                        
                            try:
                                results_dict["model"].append(model)
                            except KeyboardInterrupt:
                                interrupted = True
                            try:
                                results_dict["rep"].append(rep)
                            except KeyboardInterrupt:
                                interrupted = True
                            try:
                                results_dict["n_neighbors"].append(n_neighbors)
                            except KeyboardInterrupt:
                                interrupted = True
                            try:
                                results_dict["reconstruction_beta"].append(reconstruction_beta)
                            except KeyboardInterrupt:
                                interrupted = True
                            try:
                                results_dict["contrastive_alpha"].append(contrastive_alpha)
                            except KeyboardInterrupt:
                                interrupted = True
                            if False:
                                try:
                                    results_dict["louvain_r"].append(results_louvain[0][0])
                                except KeyboardInterrupt:
                                    interrupted = True
                                try:
                                    results_dict["louvain_mod_nx"].append(results_louvain[0][1])
                                except KeyboardInterrupt:
                                    interrupted = True
                                try:
                                    results_dict["louvain_ami"].append(results_louvain[0][2])
                                except KeyboardInterrupt:
                                    interrupted = True
                                try:
                                    results_dict["louvain_ari"].append(results_louvain[0][3])
                                except KeyboardInterrupt:
                                    interrupted = True
                            try:
                                results_dict["leiden_r"].append(results_leiden[0][0])
                            except KeyboardInterrupt:
                                interrupted = True
                            try:
                                results_dict["leiden_mod_ig"].append(results_leiden[0][1])
                            except KeyboardInterrupt:
                                interrupted = True
                            try:
                                results_dict["leiden_mod_nx"].append(results_leiden[0][2])
                            except KeyboardInterrupt:
                                interrupted = True
                            try:
                                results_dict["leiden_ami"].append(results_leiden[0][3])
                            except KeyboardInterrupt:
                                interrupted = True
                            try:
                                results_dict["leiden_ari"].append(results_leiden[0][4])
                            except KeyboardInterrupt:
                                interrupted = True
                        except KeyboardInterrupt:
                            interrupted = True
                        except Exception as e:
                            counter.write(f"ERROR {e}\n")
                            counter.update(1/NUM_REPS/len(alphas_options)/len(betas_options)/len(KNN_KS))
                            continue
                        counter.update(1/NUM_REPS/len(alphas_options)/len(betas_options)/len(KNN_KS))
                        if interrupted: break
        if interrupted: break

#%%
for e in ignored_errors:
    print(e)


#%%
results_df = pd.DataFrame(results_dict)
results_fname = f"kotliarov2020-results-ablation.csv"
if not os.path.exists(results_fname):
    results_df.to_csv(results_fname)
else:
    print("THE FILE ALREADY EXISTS, DIDN'T SAVE IT!")
results_df

# %%
ablation_df = results_df.drop(results_df[results_df["n_neighbors"]==30].index)[(
    (results_df["n_neighbors"]==15)
    &
    ~(
        (results_df["reconstruction_beta"]==0)
        & (results_df["contrastive_alpha"]==0)
    )
    & 
    ~(
        (results_df["reconstruction_beta"]==10)
        & (results_df["contrastive_alpha"]==0)
    )
    )].groupby(["reconstruction_beta", "contrastive_alpha"])[["leiden_ami","leiden_ari"]].mean()
ablation_df
#%%
f = ablation_df.plot(kind="bar")
sns.move_legend(f, "center", bbox_to_anchor=(1.2,0.5))
#%%
ablation_with_bars_df = results_df.drop(results_df[results_df["n_neighbors"]==30].index)[(
    (results_df["n_neighbors"]==15)
    &
    ~(
        (results_df["reconstruction_beta"]==0)
        & (results_df["contrastive_alpha"]==0)
    )
    & 
    ~(
        (results_df["reconstruction_beta"]==10)
        & (results_df["contrastive_alpha"]==0)
    )
    )]
ablation_with_bars_df["$\\alpha,\\beta$"] = ablation_with_bars_df["contrastive_alpha"].astype("str") + "," + ablation_with_bars_df["reconstruction_beta"].astype("str")
ablation_with_bars_df = ablation_with_bars_df.rename(columns={"leiden_ami":"Adjusted Mutual Information","leiden_ari":"Adjusted Rand Index"}).melt(["$\\alpha,\\beta$"],["Adjusted Mutual Information","Adjusted Rand Index"],"Score","Value")
f = sns.barplot(data=ablation_with_bars_df, y="Value", x="Score", hue="$\\alpha,\\beta$")
sns.move_legend(f, "center", bbox_to_anchor=(1.1,0.5))
plt.savefig("ablation.png", bbox_inches="tight")
plt.savefig("ablation.pdf", bbox_inches="tight")

# %%
knn_df = results_df.drop(results_df[results_df["n_neighbors"]==30].index)[( 
    (
        (results_df["reconstruction_beta"]==10)
        & (results_df["contrastive_alpha"]==1)
    )
    )].groupby(["n_neighbors"])[["leiden_ami","leiden_ari"]].mean()
knn_df
#%%
knn_df.plot(kind="line")
plt.savefig("knn.png", bbox_inches="tight")
plt.savefig("knn.pdf", bbox_inches="tight")
# %%
results_df.drop(results_df[results_df["n_neighbors"]==30].index)
# %%
results_df.d