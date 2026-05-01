# %%
import os
import traceback

from tqdm.autonotebook import tqdm

import numpy as np
import scipy as sp
import scipy.stats as sps
import pandas as pd

import scanpy as sc

import sklearn

import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt

from nemo.graph import get_leiden, get_louvain


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

models = [
    "totalvi", "totalviRNA", "totalviPRO",
    "subsetcontrastive", "subsetcontrastiveRNA", "subsetcontrastivePRO",
    "MANE",
]
models_with_subfolders = {
    "MANE": lambda basefolder: os.path.join(basefolder, "cmp", "mane", "kotliarov2020", "emb_{rep}.npy"),
}

resolution_resolution = 20
resolutions = np.linspace(0,2,resolution_resolution+1,endpoint=True)[1:]
resolutions

# %%
try:
    results_df = pd.read_csv("kotliarov2020-results.csv")
    results_dict = results_df.to_dict("list")
    if "Unnamed: 0" in results_dict: del results_dict["Unnamed: 0"]
except FileNotFoundError:
    results_dict = {
        "model": [],
        "rep": [],
        "louvain_r": [],
        "louvain_mod_nx": [],
        "louvain_ami": [],
        "louvain_ari": [],
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
            try:
                if (model, rep) in existing_model_rep_pairs:
                    counter.write(f"{model} {rep} already calculated before, skipping")
                    continue
                counter.set_description(f"{model} {rep}/{NUM_REPS}")
                if model not in models_with_subfolders:
                    anndata = sc.read_h5ad(os.path.join(DATA_PATH,RESULTS_PATH,f"kotliarov2020-{model}-{rep}.h5ad"))
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

                sc.pp.neighbors(anndata, use_rep='X')
                g = nx.Graph(anndata.obsp["connectivities"])

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
                counter.update(1/NUM_REPS)
                continue
            counter.update(1/NUM_REPS)
            if interrupted: break
        if interrupted: break

#%%
for e in ignored_errors:
    print(e)

#%%
NUM_SPLITS = 8
existing_model_rep_pairs = set(zip(results_dict["model"], results_dict["rep"]))
with tqdm(total=1,) as counter:
    interrupted = False
    model = "MANE-sup"
    for rep in range(5):
        try:
            if (model, rep) in existing_model_rep_pairs:
                counter.write(f"{model} {rep} already calculated before, skipping")
                continue
            emb_fpath = os.path.join(DATA_PATH,RESULTS_PATH, "cmp", "mane-sup", "kotliarov2020", "sup_emb_{rep}_{split_idx}.npy")
            pred_fpath = os.path.join(DATA_PATH,RESULTS_PATH, "cmp", "mane-sup", "kotliarov2020", "sup_pred_{rep}_{split_idx}.npy")
            tr_split_idx_fpath = os.path.join(DATA_PATH,RESULTS_PATH, "cmp", "mane-sup", "kotliarov2020", "input", "train_idx_{rep}_{split_idx}.txt")
            te_split_idx_fpath = os.path.join(DATA_PATH,RESULTS_PATH, "cmp", "mane-sup", "kotliarov2020", "input", "test_indices_{rep}_{split_idx}.txt")
            tr_label_fpath = os.path.join(DATA_PATH,RESULTS_PATH, "cmp", "mane-sup", "kotliarov2020", "input", "train_label_{rep}_{split_idx}.txt")
            te_label_fpath = os.path.join(DATA_PATH,RESULTS_PATH, "cmp", "mane-sup", "kotliarov2020", "input", "test_label_{rep}_{split_idx}.txt")
            split_idx = 0
            base_anndata = sc.AnnData(np.load(emb_fpath.format(rep=rep, split_idx=split_idx)))
            first_split_pred_with_train = np.load(pred_fpath.format(rep=rep, split_idx=split_idx))
            base_pred = np.load(pred_fpath.format(rep=rep, split_idx=split_idx))
            base_anndata.obs["MANE_Predicted_Cluster_Train_and_Test"] = np.copy(base_pred)
            idxs_explored = set()
            for split_idx in range(NUM_SPLITS):
                anndata = sc.AnnData(np.load(emb_fpath.format(rep=rep, split_idx=split_idx)))
                pred = np.load(pred_fpath.format(rep=rep, split_idx=split_idx))
                test_idxs = np.loadtxt(te_split_idx_fpath.format(rep=rep, split_idx=split_idx), dtype=int)
                train_idxs = np.loadtxt(tr_split_idx_fpath.format(rep=rep, split_idx=split_idx), dtype=int)
                test_labels = np.loadtxt(te_label_fpath.format(rep=rep, split_idx=split_idx), dtype=int)
                train_labels = np.loadtxt(tr_label_fpath.format(rep=rep, split_idx=split_idx), dtype=int)
                #base_anndata.X[test_idxs] = anndata.X[test_idxs]
                base_pred[test_idxs] = pred[test_idxs]
                base_anndata.obs[f"MANE_Label_TrainTest_{rep}_{split_idx}"] = np.concatenate([train_labels,test_labels])
                base_anndata.obs[f"MANE_Idx_TrainTest_{rep}_{split_idx}"] = np.concatenate([train_idxs,test_idxs])
                base_anndata.obs[f"MANE_Label_{rep}_{split_idx}"] = np.full_like(base_pred, -1)
                base_anndata.obs.loc[base_anndata.obs.index[train_idxs],f"MANE_Label_{rep}_{split_idx}"] = train_labels
                base_anndata.obs.loc[base_anndata.obs.index[test_idxs],f"MANE_Label_{rep}_{split_idx}"] = test_labels
                idxs_explored.update(test_idxs)
            
            if len(idxs_explored) < base_anndata.X.shape[0]:
                raise ValueError("Missing indexes!")

            if "cell_type" not in anndata.obs:
                base_anndata.obs["cell_type"] = sc.read_h5ad(os.path.join(DATA_PATH,SCRNA_FNAME)).obs["cell_type"].values
                labelmapping = dict(enumerate(sorted(base_anndata.obs["cell_type"].unique())))
                lmapf = lambda x: labelmapping[x]
                # Confirm that the indexes align
                for split_idx in range(NUM_SPLITS):
                    assert (base_anndata.obs[f"MANE_Label_{rep}_{split_idx}"].apply(lmapf) == base_anndata.obs["cell_type"]).all(), f"Rep {rep} and split {split_idx} do not match MANE- and true labels!"

            base_anndata.obs["MANE_Predicted_Cluster_Test"] = base_pred

            sc.pp.neighbors(base_anndata, use_rep='X')
            g = nx.Graph(base_anndata.obsp["connectivities"])

            counter.set_description(f"{model} {rep}/{NUM_REPS} louvain")
            clus_louvain_dict = {
                res: get_louvain(g, resolution=res)
                for res in resolutions
            }
            results_louvain = sorted(
                [
                    (r,
                    nx.community.modularity(g,clus_louvain_dict[r][0]),
                    sklearn.metrics.adjusted_mutual_info_score(base_anndata.obs["cell_type"], clus_louvain_dict[r][1]),
                    sklearn.metrics.adjusted_rand_score(base_anndata.obs["cell_type"], clus_louvain_dict[r][1]),
                    ) for r in resolutions
                ],
                key=lambda x:x[1],
                reverse=True
            )

            counter.set_description(f"{model} {rep}/{NUM_REPS} leiden")
            clus_leiden_dict = {
                res: get_leiden(base_anndata.obsp["connectivities"], resolution=res)
                for res in resolutions
            }
            results_leiden = sorted(
                [
                    (r,
                    clus_leiden_dict[r].modularity,
                    nx.community.modularity(g,[set((i for i,c in enumerate(clus_leiden_dict[r].membership) if c==k)) for k in sorted(set(clus_leiden_dict[r].membership))]),
                    sklearn.metrics.adjusted_mutual_info_score(base_anndata.obs["cell_type"], clus_leiden_dict[r].membership),
                    sklearn.metrics.adjusted_rand_score(base_anndata.obs["cell_type"], clus_leiden_dict[r].membership),
                    ) for r in resolutions
                ],
                key=lambda x:x[1],
                reverse=True
            )
            # Check that Leiden's internal modularity calculation give an equal ranking to networkx's
            rla = np.array(results_leiden)
            counter.write(f"{np.mean((rla[:,1:2]>rla[:,1:2].T)==(rla[:,2:3]>rla[:,2:3].T)):.2%} of ranking match")
            counter.set_description(f"{model} {rep}/{NUM_REPS} wrapup")
        
            results_dict["model"].append(model)
            results_dict["rep"].append(rep)
            results_dict["louvain_r"].append(results_louvain[0][0])
            results_dict["louvain_mod_nx"].append(results_louvain[0][1])
            results_dict["louvain_ami"].append(results_louvain[0][2])
            results_dict["louvain_ari"].append(results_louvain[0][3])
            results_dict["leiden_r"].append(results_leiden[0][0])
            results_dict["leiden_mod_ig"].append(results_leiden[0][1])
            results_dict["leiden_mod_nx"].append(results_leiden[0][2])
            results_dict["leiden_ami"].append(results_leiden[0][3])
            results_dict["leiden_ari"].append(results_leiden[0][4])

            results_dict["model"].append(model + "-test-pred")
            results_dict["rep"].append(rep)
            results_dict["louvain_r"].append(np.nan)
            results_dict["louvain_mod_nx"].append(np.nan)
            results_dict["louvain_ami"].append(sklearn.metrics.adjusted_mutual_info_score(base_anndata.obs["cell_type"], base_anndata.obs["MANE_Predicted_Cluster_Test"]))
            results_dict["louvain_ari"].append(sklearn.metrics.adjusted_rand_score(base_anndata.obs["cell_type"], base_anndata.obs["MANE_Predicted_Cluster_Test"]))
            results_dict["leiden_r"].append(np.nan)
            results_dict["leiden_mod_ig"].append(np.nan)
            results_dict["leiden_mod_nx"].append(np.nan)
            results_dict["leiden_ami"].append(np.nan)
            results_dict["leiden_ari"].append(np.nan)

            results_dict["model"].append(model + "-train-pred")
            results_dict["rep"].append(rep)
            results_dict["louvain_r"].append(np.nan)
            results_dict["louvain_mod_nx"].append(np.nan)
            results_dict["louvain_ami"].append(sklearn.metrics.adjusted_mutual_info_score(base_anndata.obs["cell_type"], base_anndata.obs["MANE_Predicted_Cluster_Train_and_Test"]))
            results_dict["louvain_ari"].append(sklearn.metrics.adjusted_rand_score(base_anndata.obs["cell_type"], base_anndata.obs["MANE_Predicted_Cluster_Train_and_Test"]))
            results_dict["leiden_r"].append(np.nan)
            results_dict["leiden_mod_ig"].append(np.nan)
            results_dict["leiden_mod_nx"].append(np.nan)
            results_dict["leiden_ami"].append(np.nan)
            results_dict["leiden_ari"].append(np.nan)

        except KeyboardInterrupt:
            interrupted = True
        except Exception as e:
            counter.write(f"ERROR {e}\n")
            counter.update(1/NUM_REPS)
            continue
        counter.update(1/NUM_REPS)
        if interrupted:
            counter.write("INTERRUPTED!")
            break

#%%
results_df = pd.DataFrame(results_dict)
results_fname = f"kotliarov2020-results.csv"
if not os.path.exists(results_fname):
    results_df.to_csv(results_fname)
else:
    print("THE FILE ALREADY EXISTS, DIDN'T SAVE IT!")
results_df

# %%
results_df.groupby("model").mean()
# %%
results_df.groupby("model").std()

# %%
#results_df.groupby("model").apply(lambda x: f"{np.mean(x,axis=0):.4f}±{np.std(x,axis=0):.4f}")
results_df.groupby("model").agg(["mean", "std"])

# %%
#sns.displot(, hue="model", col=["louvain_r", "louvain_mod_nx", "louvain_ami", "louvain_ari", "leiden_r", "leiden_mod_nx", "leiden_mod_ig", "leiden_ami", "leiden_ari"])
plot_df = results_df.drop(columns=["rep", "louvain_mod_nx", "leiden_mod_nx", "leiden_mod_ig"]).melt(id_vars="model")
plot_df

# %%
sns.displot(plot_df[plot_df["model"].str.startswith("totalvi")], y="value", hue="model", row="variable", kind="hist", multiple="dodge")

# %%
metric = "louvain_ami"
cmp_table = {}
for i in range(len(models)):
    m1 = models[i]
    cmp_table[(m1,m1)] = "-"
    for j in range(i+1,len(models)):
        m2 = models[j]
        x1 = results_df.loc[results_df["model"]==m1, metric]
        x2 = results_df.loc[results_df["model"]==m2, metric]
        kres = sps.kruskal(x1,x2)
        if kres.pvalue > 0.05:
            cmp_table[(m1,m2)] = "="
            cmp_table[(m2,m1)] = "="
        else:
            cmp_table[(m1,m2)] = "<" if x1.mean() < x2.mean() else ">"
            cmp_table[(m2,m1)] = "<" if x2.mean() < x1.mean() else ">"
        print(m1,m2, kres.statistic, kres.pvalue)

# %%
for metric in ["louvain_ami","louvain_ari","leiden_ami","leiden_ari"]:
    cmp_table_dict = {"model":[],**{m:[] for m in models}}
    for i in range(len(models)):
        m1 = models[i]
        cmp_table_dict["model"].append(m1)
        for j in range(len(models)):
            m2 = models[j]
            x1 = results_df.loc[results_df["model"]==m1, metric]
            x2 = results_df.loc[results_df["model"]==m2, metric]
            kres = sps.ttest_rel(x1,x2)
            if i==j:
                cmp_table_dict[m2].append("-")
                continue
            elif kres.pvalue > 0.05:
                cmp_table_dict[m2].append("=")
            else:
                cmp_table_dict[m2].append(
                    ("<" if x1.mean() < x2.mean() else ">")
                    + "*" * int(np.floor(-np.log10(kres.pvalue)))
                )
            if j>i:
                print(m1,m2, kres.statistic, kres.pvalue)
    cmp_table_df = pd.DataFrame(cmp_table_dict).set_index("model")
    cmp_table_df = cmp_table_df.style.set_table_attributes("style='display:inline'").set_caption(metric.replace("_", " ").title())
    try:
        display(cmp_table_df)
    except NameError:
        print(cmp_table_df)
# %%
# %%
for metrics in [["louvain_ami","leiden_ami",],["louvain_ari","leiden_ari"]]:
    cmp_table_dict = {"model":[],**{(mod,met):[] for mod in models for met in metrics}}
    for i in range(len(models)):
        for met1 in metrics:
            mod1 = models[i]
            cmp_table_dict["model"].append((mod1,met1))
            for j in range(len(models)):
                for met2 in metrics:
                    mod2 = models[j]
                    x1 = results_df.loc[results_df["model"]==mod1, met1]
                    x2 = results_df.loc[results_df["model"]==mod2, met2]
                    kres = sps.ttest_rel(x1,x2)
                    if i==j and met1==met2:
                        cmp_table_dict[(mod2,met2)].append("-")
                        continue
                    elif kres.pvalue > 0.05:
                        cmp_table_dict[(mod2,met2)].append("=")
                    else:
                        cmp_table_dict[(mod2,met2)].append(
                            ("<" if x1.mean() < x2.mean() else ">")
                            + "*" * int(np.floor(-np.log10(kres.pvalue)))
                        )
                    if j>i:
                        print((mod1,met1), (mod2,met2), kres.statistic, kres.pvalue)
    cmp_table_df = pd.DataFrame(cmp_table_dict).set_index("model")
    cmp_table_df = cmp_table_df.style.set_table_attributes("style='display:inline'").set_caption(str(metrics))
    try:
        display(cmp_table_df)
    except NameError:
        print(cmp_table_df)

# %%
