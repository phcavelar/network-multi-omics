# %%
import os

from tqdm.autonotebook import tqdm

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
layers = ["miRNA", "protein", "rna",]# "prot"]
fextension = ["fea", "fea", "fea",]# "csv"]

NUM_REPS = 8

# %%
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
            # To compare with others
            #os.makedirs(f"amerged/{cancer}/", exist_ok=True)
            #data[data_type].to_csv(f"amerged/{cancer}/{data_type.upper() if data_type=='rna' else data_type}.fea", sep="\t")
            #continue
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
        #continue
        for repetition in range(NUM_REPS):
            if os.path.exists(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-rnapro-{repetition}.h5ad")):
                counter.update()
                continue
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
            adz.write(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-rnapro-{repetition}.h5ad"))
            os.makedirs(os.path.join(DATA_PATH,MODEL_DIR), exist_ok=True)
            gae.save(os.path.join(DATA_PATH,MODEL_DIR,f"tcga-{cancer}-subsetcontrastive-rnapro-{repetition}.model"))
            history_df = pd.DataFrame(tr_log)
            history_df.to_csv(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-rnapro-{repetition}-history.csv.gz"))
            counter.update()

# %%
# Load first model for diagnostic plots
for cancer in ["BRCA"]:#TCGA_cancers:
    z = sc.read_h5ad(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-rnapro-0.h5ad"))
    history_df = pd.read_csv(os.path.join(DATA_PATH,RESULTS_PATH,f"tcga-{cancer}-subsetcontrastive-rnapro-0-history.csv.gz"))
    subtype_cols = subtype_cols_per_cancer[cancer]
    stage_col = stage_col_per_cancer[cancer]
    clin_cols = ['gender', *subtype_cols, stage_col]

    sc.pp.neighbors(z, use_rep="X")
    sc.tl.umap(z)
    sc.pl.umap(z, color=clin_cols, ncols=1)
    plt.show()
    
    break
    history_plot_keys = [
        ['total', 'r', 'c'],
        ['r', 'r_0', 'r_1',],
        ['c', 'c1'],# 'c2',],
        ['c1', 'c1_0', 'c1_1',],
        #['c2', 'c2_0', 'c2_1',],
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
for cancer in ["BRCA"]:#TCGA_cancers:
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
        if False or False:
            print(cancer, clin.shape[0], sep="\t", file=counter)
        for data_type in data:
            data[data_type] = data[data_type].loc[index_in_all]
            # To compare with others
            #os.makedirs(f"amerged/{cancer}/", exist_ok=True)
            #data[data_type].to_csv(f"amerged/{cancer}/{data_type.upper() if data_type=='rna' else data_type}.fea", sep="\t")
            #continue
            data[data_type] = sc.AnnData(data[data_type], obs=clin.loc[data[data_type].index,clin_cols])
            data[data_type].X = data[data_type].X.astype(np.float32)
            sc.pp.neighbors(data[data_type])
            if False:
                print("",data_type, data[data_type].shape[1], f"{np.mean(data[data_type]):.2f}", f"{np.std(data[data_type].values):.2f}", sep="\t", file=counter)
                print("","","feat-wise", f"{np.mean(np.mean(data[data_type],axis=0)):.2f}", f"{np.mean(np.std(data[data_type],axis=0)):.2f}", sep="\t", file=counter)
                print("","","sample-wise", f"{np.mean(np.mean(data[data_type],axis=1)):.2f}", f"{np.mean(np.std(data[data_type],axis=1)):.2f}", sep="\t", file=counter)
            if True and data_type in data:
                print(data_type)
                sc.tl.umap(data[data_type])
                sc.pl.umap(data[data_type], color=['gender', *subtype_cols, stage_col],)
                plt.gcf().suptitle(f"{cancer} {data_type}")
                plt.show()
# %%
