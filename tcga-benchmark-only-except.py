# %%
import os
import shutil
import warnings

import lifelines.statistics
import sklearn.metrics
from tqdm.autonotebook import tqdm

import numpy as np
import scipy as sp
import scipy.stats as sps
import pandas as pd

import scanpy as sc

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
import sklearn.cluster

import lifelines

from nemo.models.subsetcontrastive import SubsetContrastive
from nemo.graph import get_leiden

import torch
import torch.nn as nn

# %%
DATA_PATH = "~/data/netemo"
DATA_PATH = os.path.expanduser(os.path.expandvars(DATA_PATH))
MODEL_DIR = "models"
RESULTS_PATH = "results"

TCGA_PATH = "~/data/subtypemgtp"
TCGA_PATH = os.path.expanduser(os.path.expandvars(TCGA_PATH))
TCGA_cancers = ['BLCA', 'BRCA',
                #'GBM', #no gbm prot.fea
                'KIRC', 'LUAD',
                #'PAAD', #no paad prot.fea and too small
                'SKCM', 'STAD',
                'UCEC',
                #'UVM', #uvm is too small
            ]
subtype_cols_per_cancer = {
    "BLCA": ["diagnosis_subtype",],
    "BRCA": ["PAM50Call_RNAseq",],
    "GBM": ["GeneExp_Subtype",],
    "KIRC": [],
    "LUAD": ["Expression_Subtype",],
    "PAAD": [],
    "SKCM": ["sample_type",],
    "STAD": ["barretts_esophagus",],
    "UCEC": ['diabetes', 'histological_type',],
    "UVM": ["tumor_shape_pathologic_clinical",],
}
stage_col_per_cancer = {
    **{
        c: "pathologic_stage"
        for c in TCGA_cancers
    },
    "UCEC": "clinical_stage"
}
layers = ["CN", "meth", "miRNA", "protein", "rna",]# "prot"]
fextension = ["fea", "fea", "fea", "fea", "fea",]# "csv"]

NUM_REPS = 8

# %%
"""
print_feature_n_samples = False
print_feature_distributions = False
plot_target_distributions = False

with tqdm(total=len(TCGA_cancers)*NUM_REPS) as counter:

    for cancer in TCGA_cancers:
        subtype_cols = subtype_cols_per_cancer[cancer]
        stage_col = stage_col_per_cancer[cancer]
        clin_cols = ['gender', *subtype_cols, stage_col]

        data = {}
        clin = pd.read_csv(os.path.join(TCGA_PATH,cancer,f"clinical.tsv"), index_col=0, sep="\t")
        clin = clin.sort_index()
        # Gets only the primary tumour sample metadata
        clin = clin.loc[~clin.index.str.split("-").str[:-1].str.join("-").duplicated(keep='first')]
        # Make it patient_id instead of tumour_id
        clin = clin.set_index(clin.index.str.split("-").str[:-1].str.join("-"))

        index_in_all = set(clin.index)
        for data_type, ftype in zip(layers,fextension):
            try:
                try:
                    match ftype:
                        case "fea":
                            data[data_type] = pd.read_csv(os.path.join(TCGA_PATH,cancer,f"{data_type}.{ftype}"), index_col=0).T
                        case "csv":
                            data[data_type] = pd.read_csv(os.path.join(TCGA_PATH,cancer,f"{data_type}.{ftype}"), index_col=0).drop(columns=["Cancer_Type", "Sample_Type", "SetID"])
                except FileNotFoundError:
                    continue
                if index_in_all is None:
                    index_in_all = set(data[data_type].index.values)
                index_in_all.intersection_update(data[data_type].index.values)
            except OSError:
                raise

        index_in_all = sorted(index_in_all)
        clin = clin.loc[index_in_all]
        if print_feature_n_samples or print_feature_distributions:
            print(cancer, clin.shape[0], sep="\t", file=counter)
        for data_type in data:
            data[data_type] = data[data_type].loc[index_in_all]
            data[data_type] = sc.AnnData(data[data_type], obs=clin.loc[data[data_type].index,clin_cols])
            data[data_type].X = data[data_type].X.astype(np.float32)
            sc.pp.neighbors(data[data_type])
            if print_feature_distributions:
                print("",data_type, data[data_type].shape[1], f"{np.mean(data[data_type]):.2f}", f"{np.std(data[data_type].values):.2f}", sep="\t", file=counter)
                print("","","feat-wise", f"{np.mean(np.mean(data[data_type],axis=0)):.2f}", f"{np.mean(np.std(data[data_type],axis=0)):.2f}", sep="\t", file=counter)
                print("","","sample-wise", f"{np.mean(np.mean(data[data_type],axis=1)):.2f}", f"{np.mean(np.std(data[data_type],axis=1)):.2f}", sep="\t", file=counter)
            if plot_target_distributions and data_type in data:
                sc.tl.umap(data[data_type])
                sc.pl.umap(data[data_type], color=['gender', *subtype_cols, stage_col],)
                plt.gcf().suptitle(f"{cancer} {data_type}")
                plt.show()

        for repetition in range(NUM_REPS):
            gae = SubsetContrastive(
                data,
                {k:nn.MSELoss() for k in data},
                200,#0.2,
                num_epochs=1024,
            )
            tr_log = gae.train(data)
            Xs = [torch.tensor(data[m].X, dtype=torch.float) for m in gae.modalities]
            As = [torch.tensor(np.stack(data[m].obsp["connectivities"].nonzero()), dtype=torch.long) for m in gae.modalities]
            _, __, z = gae.model(Xs, As)

            adz = sc.AnnData(z.detach().cpu().numpy())
            adz.obs = data["rna"].obs

            os.makedirs(os.path.join(DATA_PATH,RESULTS_PATH), exist_ok=True)
            adz.write(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-{repetition}.h5ad"))
            os.makedirs(os.path.join(DATA_PATH,MODEL_DIR), exist_ok=True)
            gae.save(os.path.join(DATA_PATH,MODEL_DIR,f"tcga-{cancer}-subsetcontrastive-{repetition}.model"))
            history_df = pd.DataFrame(tr_log)
            history_df.to_csv(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-{repetition}-history.csv.gz"))
            counter.update()"""

# %%
# Load first model for diagnostic plots
#resolution_resolution = 20
#resolutions = np.linspace(0,2,resolution_resolution+1,endpoint=True)[1:]

cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,
                'LUAD': 3,'SKCM': 4, 
                'STAD': 3, 'UCEC': 4, 'UVM': 4}

def find_leiden_with_k(k,adata):
    current_r = 1.0
    while True:
        this_res = get_leiden(adata.obsp["connectivities"], current_r)
        this_k = len(this_res.sizes())
        if this_k == k:
            return this_res
        elif this_k < k:
            current_r *= 1.5
        else:
            current_r *= 0.5

def find_leiden_with_best_resolution(adata, resolutions=np.linspace(0,2,21,endpoint=True)[1:]):
    clus_leiden_dict = {
        res: get_leiden(adata.obsp["connectivities"], resolution=res)
        for res in resolutions
    }
    results_leiden = sorted(
        [
            (r,
            clus_leiden_dict[r].modularity,
            ) for r in resolutions
        ],
        key=lambda x:x[1],
        reverse=True
    )
    return clus_leiden_dict[results_leiden[0][0]]

def find_leiden_with_best_resolution_and_with_k(k, adata, resolutions=np.linspace(0,2,21,endpoint=True)[1:]):
    clus_leiden_dict = {
        res: get_leiden(adata.obsp["connectivities"], resolution=res)
        for res in resolutions
    }
    results_leiden = sorted(
        [
            (r,
            clus_leiden_dict[r].modularity,
            ) for r in resolutions
            if len(clus_leiden_dict[r].sizes())==k
        ],
        key=lambda x:x[1],
        reverse=True
    )
    if len(results_leiden)<=0:
        return find_leiden_with_k(k,adata)
    return clus_leiden_dict[results_leiden[0][0]]

results = {
    "cancer":[],
    "clusterer":[],
    "k":[],
    "repetition":[],
    "layer":[],
    "presence":[],
    "-log10p":[],
    #"ari":[],
    #"ami":[],
}

#%%

for cancer in TCGA_cancers:
    all = os.listdir(os.path.join(DATA_PATH,RESULTS_PATH))#,f"tcga-{cancer}-subsetcontrastive-only-*-{repetition}.h5ad"))
    only = sorted(
        set(
            map(
                (lambda x: x.split("only-")[1].split("-")[0]),
                filter(
                    (lambda x: "only" in x and cancer in x),
                    all
                )
            )
        )
    )
    exceptthis = sorted(
        set(
            map(
                (lambda x: x.split("except-")[1].split("-")[0]),
                filter(
                    (lambda x: "except" in x and cancer in x),
                    all
                )
            )
        )
    )
    print(cancer, only, exceptthis)
    assert len(set(only).difference(exceptthis))==0, "Only should be equal to except"

    mlog10p = []
    resolutions = []
    for repetition in range(NUM_REPS):
        z = sc.read_h5ad(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-{repetition}.h5ad"))
        #history_df = pd.read_csv(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-{repetition}-history.csv.gz"))
        subtype_cols = subtype_cols_per_cancer[cancer]
        stage_col = stage_col_per_cancer[cancer]
        clin_cols = ['gender', *subtype_cols, stage_col]
        surv = pd.read_csv(os.path.join(TCGA_PATH,cancer,f"survival.tsv"), index_col=0, sep="\t")
        surv = surv.sort_index()
        # Gets only the primary tumour sample metadata
        surv = surv.loc[~surv.index.str.split("-").str[:-1].str.join("-").duplicated(keep='first')]
        # Make it patient_id instead of tumour_id
        surv = surv.set_index(surv.index.str.split("-").str[:-1].str.join("-"))

        sc.pp.neighbors(z, use_rep="X")

        k = cancer_dict[cancer]
        for clusterer_name, cluster_this in [
                            ("kmeans", lambda: sklearn.cluster.KMeans(k).fit_predict(z.X)),
                            ("kleiden", lambda: find_leiden_with_best_resolution_and_with_k(k, z).membership),
                            ("leiden", lambda: find_leiden_with_best_resolution(z).membership),
                            ]:
            cl = cluster_this()

            clustered_survival = surv.join(pd.Series(cl,z.obs.index,name="Cluster")).dropna()
            res = lifelines.statistics.multivariate_logrank_test(clustered_survival["OS.time"],
                                                        clustered_survival["Cluster"].astype(int),
                                                        clustered_survival["OS"])
            mlog10p.append(-np.log10(res.p_value))
            results["cancer"].append(cancer)
            results["clusterer"].append(clusterer_name)
            results["k"].append(np.max(cl)+1)
            results["repetition"].append(repetition)
            results["layer"].append("all")
            results["presence"].append("all")
            results["-log10p"].append(-np.log10(res.p_value))
            #results["ari"].append(sklearn.metrics)
            #results["ami"].append(-np.log10(res.p_value))

    mlog10p = []
    resolutions = []
    for repetition in range(NUM_REPS):
        z = sc.read_h5ad(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-rnapro-{repetition}.h5ad"))
        #history_df = pd.read_csv(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-rnapro-{repetition}-history.csv.gz"))
        subtype_cols = subtype_cols_per_cancer[cancer]
        stage_col = stage_col_per_cancer[cancer]
        clin_cols = ['gender', *subtype_cols, stage_col]
        surv = pd.read_csv(os.path.join(TCGA_PATH,cancer,f"survival.tsv"), index_col=0, sep="\t")
        surv = surv.sort_index()
        # Gets only the primary tumour sample metadata
        surv = surv.loc[~surv.index.str.split("-").str[:-1].str.join("-").duplicated(keep='first')]
        # Make it patient_id instead of tumour_id
        surv = surv.set_index(surv.index.str.split("-").str[:-1].str.join("-"))

        sc.pp.neighbors(z, use_rep="X")

        k = cancer_dict[cancer]
        for clusterer_name, cluster_this in [
                            ("kmeans", lambda: sklearn.cluster.KMeans(k).fit_predict(z.X)),
                            ("kleiden", lambda: find_leiden_with_best_resolution_and_with_k(k, z).membership),
                            ("leiden", lambda: find_leiden_with_best_resolution(z).membership),
                            ]:
            cl = cluster_this()

            clustered_survival = surv.join(pd.Series(cl,z.obs.index,name="Cluster")).dropna()
            res = lifelines.statistics.multivariate_logrank_test(clustered_survival["OS.time"],
                                                        clustered_survival["Cluster"].astype(int),
                                                        clustered_survival["OS"])
            mlog10p.append(-np.log10(res.p_value))
            results["cancer"].append(cancer)
            results["clusterer"].append(clusterer_name)
            results["k"].append(np.max(cl)+1)
            results["repetition"].append(repetition)
            results["layer"].append("all")
            results["presence"].append("except")
            results["-log10p"].append(-np.log10(res.p_value))
            #results["ari"].append(sklearn.metrics)
            #results["ami"].append(-np.log10(res.p_value))
    print(cancer, "all", "except", np.mean(mlog10p), np.std(mlog10p), *[np.quantile(mlog10p,q=q) for q in [0,0.25,0.5,0.75,1.]])
    
    for onlyorexcept in ["only","except"]:
        for omics_layer in only:
            mlog10p = []
            resolutions = []
            for repetition in range(NUM_REPS):
        
                z = sc.read_h5ad(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-{onlyorexcept}-{omics_layer}-{repetition}.h5ad"))
                #history_df = pd.read_csv(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-{onlyorexcept}-{omics_layer}-{repetition}-history.csv.gz"))
                subtype_cols = subtype_cols_per_cancer[cancer]
                stage_col = stage_col_per_cancer[cancer]
                clin_cols = ['gender', *subtype_cols, stage_col]
                surv = pd.read_csv(os.path.join(TCGA_PATH,cancer,f"survival.tsv"), index_col=0, sep="\t")
                surv = surv.sort_index()
                # Gets only the primary tumour sample metadata
                surv = surv.loc[~surv.index.str.split("-").str[:-1].str.join("-").duplicated(keep='first')]
                # Make it patient_id instead of tumour_id
                surv = surv.set_index(surv.index.str.split("-").str[:-1].str.join("-"))

                sc.pp.neighbors(z, use_rep="X")

                k = cancer_dict[cancer]
                #clusterer = sklearn.cluster.KMeans(k)
                #cl = clusterer.fit_predict(z.X)
                #res_leiden = get_leiden(z.obsp["connectivities"], 1)
                #res_leiden = find_leiden_with_k(k, z)
                for clusterer_name, cluster_this in [
                                    ("kmeans", lambda: sklearn.cluster.KMeans(k).fit_predict(z.X)),
                                    ("kleiden", lambda: find_leiden_with_best_resolution_and_with_k(k, z).membership),
                                    ("leiden", lambda: find_leiden_with_best_resolution(z).membership),
                                    ]:
                    cl = cluster_this()

                    clustered_survival = surv.join(pd.Series(cl,z.obs.index,name="Cluster")).dropna()
                    res = lifelines.statistics.multivariate_logrank_test(clustered_survival["OS.time"],
                                                                clustered_survival["Cluster"].astype(int),
                                                                clustered_survival["OS"])
                    mlog10p.append(-np.log10(res.p_value))
                    results["cancer"].append(cancer)
                    results["k"].append(np.max(cl)+1)
                    results["clusterer"].append(clusterer_name)
                    results["repetition"].append(repetition)
                    results["layer"].append(omics_layer)
                    results["presence"].append(onlyorexcept)
                    results["-log10p"].append(-np.log10(res.p_value))
            print(cancer, onlyorexcept, omics_layer, np.mean(mlog10p), np.std(mlog10p), *[np.quantile(mlog10p,q=q) for q in [0,0.25,0.5,0.75,1.]])
    

#%%
# SubtypeMGTP runs

for cancer in TCGA_cancers:
    mlog10p = []
    resolutions = []
    for repetition in range(NUM_REPS):
        cl = pd.read_csv(os.path.join(DATA_PATH,RESULTS_PATH,"cmp","subtypemgtp",f"{cancer}-{repetition}.cluster"), sep="\t", index_col=0)
        surv = pd.read_csv(os.path.join(TCGA_PATH,cancer,f"survival.tsv"), index_col=0, sep="\t")
        surv = surv.sort_index()
        # Gets only the primary tumour sample metadata
        surv = surv.loc[~surv.index.str.split("-").str[:-1].str.join("-").duplicated(keep='first')]
        # Make it patient_id instead of tumour_id
        surv = surv.set_index(surv.index.str.split("-").str[:-1].str.join("-"))

        clustered_survival = surv.join(cl).dropna()
        res = lifelines.statistics.multivariate_logrank_test(clustered_survival["OS.time"],
                                                    clustered_survival["cluster"].astype(int),
                                                    clustered_survival["OS"])
        mlog10p.append(-np.log10(res.p_value))
        results["cancer"].append(cancer)
        results["clusterer"].append("SubtypeMGTP")
        results["k"].append(np.max(cl)+1)
        results["repetition"].append(repetition)
        results["layer"].append("protein")
        results["presence"].append("only")
        results["-log10p"].append(-np.log10(res.p_value))
        #results["ari"].append(sklearn.metrics)
        #results["ami"].append(-np.log10(res.p_value))
    print(cancer, "subtypemgtp", "*", np.mean(mlog10p), np.std(mlog10p), *[np.quantile(mlog10p,q=q) for q in [0,0.25,0.5,0.75,1.]])

#%%
# SURER Runs
for cancer in TCGA_cancers:
    mlog10p = []
    for repetition in range(NUM_REPS):
        fpath = os.path.join(
            DATA_PATH,
            RESULTS_PATH,
            "cmp",
            "surer",
            f"SURER-tcga-{cancer}-surerinput_k{cancer_dict[cancer]}_{repetition}.cluster"
        )
        try:
            cl = pd.read_csv(
                fpath,
                sep="\t",
                index_col=0,
            )
        except FileNotFoundError:
            #print("NOT FOUND: ", fpath)
            continue
        surv = pd.read_csv(os.path.join(TCGA_PATH,cancer,f"survival.tsv"), index_col=0, sep="\t")
        surv = surv.sort_index()
        # Gets only the primary tumour sample metadata
        surv = surv.loc[~surv.index.str.split("-").str[:-1].str.join("-").duplicated(keep='first')]
        # Make it patient_id instead of tumour_id
        surv = surv.set_index(surv.index.str.split("-").str[:-1].str.join("-"))

        clustered_survival = surv.join(cl).dropna()
        res = lifelines.statistics.multivariate_logrank_test(clustered_survival["OS.time"],
                                                    clustered_survival["cluster"].astype(int),
                                                    clustered_survival["OS"])
        mlog10p.append(-np.log10(res.p_value))
        results["cancer"].append(cancer)
        results["clusterer"].append("SURER")
        results["k"].append(np.max(cl)+1)
        results["repetition"].append(repetition)
        results["layer"].append("all")
        results["presence"].append("all")
        results["-log10p"].append(-np.log10(res.p_value))
        #results["ari"].append(sklearn.metrics)
        #results["ami"].append(-np.log10(res.p_value))
    if len(mlog10p)==0:
        continue
    print(cancer, "surer", "*", np.mean(mlog10p), np.std(mlog10p), *[np.quantile(mlog10p,q=q) for q in [0,0.25,0.5,0.75,1.]])


#%%
# MANE Runs
for cancer in TCGA_cancers:
    subtype_cols = subtype_cols_per_cancer[cancer]
    stage_col = stage_col_per_cancer[cancer]
    clin_cols = ['gender', *subtype_cols, stage_col]

    data = {}
    clin = pd.read_csv(os.path.join(TCGA_PATH,cancer,f"clinical.tsv"), index_col=0, sep="\t")
    clin = clin.sort_index()
    # Gets only the primary tumour sample metadata
    clin = clin.loc[~clin.index.str.split("-").str[:-1].str.join("-").duplicated(keep='first')]
    # Make it patient_id instead of tumour_id
    clin = clin.set_index(clin.index.str.split("-").str[:-1].str.join("-"))

    index_in_all = set(clin.index)
    for data_type, ftype in zip(layers,fextension):
        try:
            try:
                match ftype:
                    case "fea":
                        data[data_type] = pd.read_csv(os.path.join(TCGA_PATH,cancer,f"{data_type}.{ftype}"), index_col=0).T
                    case "csv":
                        data[data_type] = pd.read_csv(os.path.join(TCGA_PATH,cancer,f"{data_type}.{ftype}"), index_col=0).drop(columns=["Cancer_Type", "Sample_Type", "SetID"])
            except FileNotFoundError:
                continue
            if index_in_all is None:
                index_in_all = set(data[data_type].index.values)
            index_in_all.intersection_update(data[data_type].index.values)
        except OSError:
            raise

    index_in_all = sorted(index_in_all)
    clin = clin.loc[index_in_all]
    print(cancer, clin.shape)
    
    dset_pathname = f"tcga_{cancer}"
    os.makedirs(os.path.join(dset_pathname), exist_ok=True)

    mlog10p = []
    resolutions = []

    for repetition in range(NUM_REPS):
        fpath = os.path.join(
            DATA_PATH,
            RESULTS_PATH,
            "cmp",
            "mane",
            f"tcga_{cancer}",
            f"emb_{repetition}.npy"
        )
    
        X = np.load(
            fpath
        )
        z = sc.AnnData(X, obs=clin)
        sc.pp.neighbors(z, use_rep="X")

        surv = pd.read_csv(os.path.join(TCGA_PATH,cancer,f"survival.tsv"), index_col=0, sep="\t")
        surv = surv.sort_index()
        # Gets only the primary tumour sample metadata
        surv = surv.loc[~surv.index.str.split("-").str[:-1].str.join("-").duplicated(keep='first')]
        # Make it patient_id instead of tumour_id
        surv = surv.set_index(surv.index.str.split("-").str[:-1].str.join("-"))

        k = cancer_dict[cancer]
        for clusterer_name, cluster_this in [
                            ("kmeans", lambda: sklearn.cluster.KMeans(k).fit_predict(z.X)),
                            ("kleiden", lambda: find_leiden_with_best_resolution_and_with_k(k, z).membership),
                            ("leiden", lambda: find_leiden_with_best_resolution(z).membership),
                            ]:
            cl = cluster_this()

            clustered_survival = surv.join(pd.Series(cl,z.obs.index,name="Cluster")).dropna()
            res = lifelines.statistics.multivariate_logrank_test(clustered_survival["OS.time"],
                                                        clustered_survival["Cluster"].astype(int),
                                                        clustered_survival["OS"])
            mlog10p.append(-np.log10(res.p_value))
            results["cancer"].append(cancer)
            results["clusterer"].append(f"MANE-{clusterer_name}")
            results["k"].append(np.max(cl)+1)
            results["repetition"].append(repetition)
            results["layer"].append("all")
            results["presence"].append("all")
            results["-log10p"].append(-np.log10(res.p_value))
    if len(mlog10p)==0:
        continue
    print(cancer, "mane", "*", np.mean(mlog10p), np.std(mlog10p), *[np.quantile(mlog10p,q=q) for q in [0,0.25,0.5,0.75,1.]])


#%%
results_df = pd.DataFrame(results)
results_fname = f"tcga-results.csv"
if not os.path.exists(results_fname):
    results_df.to_csv(results_fname)
else:
    print("THE FILE ALREADY EXISTS, DIDN'T SAVE IT!")
results_df

# %%
fg = sns.FacetGrid(results_df, row="cancer", col="clusterer", hue="presence", sharey="row")
fg.map_dataframe(sns.swarmplot, x="layer", y="-log10p", size=2)
fg.refline(y=-np.log10(0.05))
#row=None, col=None, hue=None, col_wrap=None, sharex=True, sharey=True, height=3, aspect=1, palette=None, row_order=None, col_order=None, hue_order=None, hue_kws=None, dropna=False, legend_out=True, despine=True, margin_titles=False, xlim=None, ylim=None, subplot_kws=None, gridspec_kws=None)
#for cancer in TCGA_cancers:
#    sns.swarmplot(results_df[results_df["cancer"]==cancer], x="layer", y="-log10p", hue="presence")
#    plt.plot([-0.5,5.5],[-np.log10(0.05)]*2, c="k", linestyle='dashed')
#    plt.title(cancer)
#    plt.show()
# %%
results_df.loc[
    (
        (results_df["presence"]=="only")&(results_df["layer"]=="rna")&(results_df["clusterer"]=="kleiden")
        |
        (results_df["presence"]=="only")&(results_df["layer"]=="protein")&(results_df["clusterer"]=="SubtypeMGTP")
        |
        (results_df["presence"]=="all")&(results_df["layer"]=="all")&(results_df["clusterer"]=="SURER")
    )
    ].groupby(["cancer","layer","presence","clusterer"])["-log10p"].mean()
# %%
results_df.loc[
    (
        (results_df["presence"]=="only")&(results_df["layer"]=="protein")&(results_df["clusterer"]=="kleiden")
        |
        (results_df["presence"]=="only")&(results_df["layer"]=="protein")&(results_df["clusterer"]=="SubtypeMGTP")
        |
        (results_df["presence"]=="all")&(results_df["layer"]=="all")&(results_df["clusterer"]=="SURER")
    )
    ].groupby(["cancer","layer","presence","clusterer"])["-log10p"].mean()
# %%
results_df.loc[
    (
        (results_df["presence"]=="all")&(results_df["layer"]=="all")&(results_df["clusterer"]=="kleiden")
        |
        (results_df["presence"]=="only")&(results_df["layer"]=="protein")&(results_df["clusterer"]=="SubtypeMGTP")
        |
        (results_df["presence"]=="all")&(results_df["layer"]=="all")&(results_df["clusterer"]=="SURER")
    )
    ].groupby(["cancer","layer","presence","clusterer"])["-log10p"].mean()
# %% THIS ONE
results_df.loc[
    (
        (results_df["presence"]=="except")&(results_df["layer"]=="all")&(results_df["clusterer"]=="kleiden")
        |
        (results_df["presence"]=="only")&(results_df["layer"]=="protein")&(results_df["clusterer"]=="SubtypeMGTP")
        |
        (results_df["presence"]=="all")&(results_df["layer"]=="all")&(results_df["clusterer"]=="SURER")
        |
        #(results_df["presence"]=="all")&(results_df["layer"]=="all")&(results_df["clusterer"].str.startswith("MANE-"))&(~results_df["clusterer"].str.endswith("-leiden"))
        (results_df["presence"]=="all")&(results_df["layer"]=="all")&(results_df["clusterer"]=="MANE-kleiden")
        #(results_df["presence"]=="all")&(results_df["layer"]=="all")&(results_df["clusterer"]=="MANE-kmeans")
    )
    ].groupby(["cancer","layer","presence","clusterer"])["-log10p"].agg(["mean","std","max"])
# %%
results_df.loc[
    (
        (results_df["clusterer"]=="kleiden")
        |
        (results_df["clusterer"]=="SubtypeMGTP")
        |
        (results_df["clusterer"]=="SURER")
    )
    ].groupby(["cancer","layer","presence","clusterer"])["-log10p"].agg(["mean","std","max"])
# %%
results_df.loc[(results_df["clusterer"]=="kleiden")].groupby(["cancer","layer","presence","clusterer"])["-log10p"].agg(["mean","std","max"])
# %%
results_df.loc[
(
    ((
        ((results_df["presence"]=="except")&(results_df["layer"]=="all"))
        |((results_df["presence"]=="only")&(results_df["presence"]!="CN")&(results_df["presence"]!="meth"))
    )&(results_df["clusterer"]=="kleiden"))
    |
    (results_df["presence"]=="only")&(results_df["layer"]=="protein")&(results_df["clusterer"]=="SubtypeMGTP")
)
].groupby(["cancer","layer","presence","clusterer"])["-log10p"].agg(lambda x: f"${x.mean():.2f}\\pm{x.std():.2f}$ ({x.max():.2f})")
# %%
print(
    results_df.loc[
    (
        (((results_df["presence"]=="except")&(results_df["layer"]=="all"))|((results_df["presence"]=="only")&(results_df["presence"]!="CN")&(results_df["presence"]!="meth")))&(results_df["clusterer"]=="kleiden")
        |
        (results_df["presence"]=="only")&(results_df["layer"]=="protein")&(results_df["clusterer"]=="SubtypeMGTP")
    )
    ].groupby(["cancer","layer","presence","clusterer"])["-log10p"].agg(lambda x: f"${x.mean():.2f}\\pm{x.std():.2f}$ ({x.max():.2f})").reset_index([1,2,3]).drop(columns=["presence"]).rename(columns={"clusterer":"model"}).pivot(columns=["layer","model"]).T.reset_index().drop(columns=["level_0"]).set_index("layer").replace("kleiden","$SC^{2}$").to_latex()
)
    
# %%
for cancer in TCGA_cancers:
    sc2both = results_df.loc[
        (
            (results_df["cancer"] == cancer)
            & (results_df["presence"]=="except")
            & (results_df["layer"]=="all")
            & (results_df["clusterer"]=="kleiden")
        ),
        "-log10p"
    ].values
    subtypemgtp = results_df.loc[
        (
            (results_df["cancer"] == cancer)
            & (results_df["presence"]=="only")
            & (results_df["layer"]=="protein")
            & (results_df["clusterer"]=="SubtypeMGTP")
        ),
        "-log10p"
    ].values
    res = sps.ttest_ind(sc2both, subtypemgtp)
    print(cancer, "mgtp", res.pvalue<0.05, res.pvalue)

for cancer in TCGA_cancers:
    vals = [
        ("MGTP ","only","protein","SubtypeMGTP"),
        ("SURER ","all","all","SURER"),
        ("SC2  ","except","all","kleiden"),
        ("RNA  ","only","rna","kleiden"),
        ("prot ","only","protein","kleiden"),
        ("miRNA","only","miRNA","kleiden"),
        ("meth ","only","meth","kleiden"),
        ("CN   ","only","CN","kleiden"),
    ]
    res_table = []
    for i in range(len(vals)):
        res_table.append([])
        vi = results_df.loc[
            (
                (results_df["cancer"] == cancer)
                & (results_df["presence"]==vals[i][1])
                & (results_df["layer"]==vals[i][2])
                & (results_df["clusterer"]==vals[i][3])
            ),
            "-log10p"
        ].values
        for j in range(len(vals)):
            vj = results_df.loc[
                (
                    (results_df["cancer"] == cancer)
                    & (results_df["presence"]==vals[j][1])
                    & (results_df["layer"]==vals[j][2])
                    & (results_df["clusterer"]==vals[j][3])
                ),
                "-log10p"
            ].values

            res = sps.ttest_ind(vi, vj)

            if res.pvalue>=0.05:
                res_table[i].append(f"≅")
            else:
                res_table[i].append(f"<" if vi.mean()<vj.mean() else f">")
    print(pd.DataFrame(res_table, columns=[v[0] for v in vals], index=pd.Index([v[0] for v in vals], name=cancer)))
# %%
