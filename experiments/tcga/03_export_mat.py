# %%
import os

from tqdm.autonotebook import tqdm

import numpy as np
import scipy as sp
import scipy.io as spio
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
layers = ["CN", "meth", "miRNA", "protein", "rna",]# "prot"]
fextension = ["fea", "fea", "fea", "fea", "fea",]# "csv"]

cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,
                'LUAD': 3,'SKCM': 4, 
                'STAD': 3, 'UCEC': 4, 'UVM': 4}

NUM_REPS = 8

# %%
print_feature_n_samples = False
print_feature_distributions = False
plot_target_distributions = False

with tqdm(total=len(TCGA_cancers)) as counter:

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

        placeholder_labels = np.random.randint(cancer_dict[cancer],size=(1,data[sorted(data.keys())[0]].X.shape[0]))
        
        surer_mat_data = {
            "truelabel": np.array(
                [[None for layer in sorted(data.keys())]],
                dtype=np.object_
            ),
            "data": np.array(
                [[None for layer in sorted(data.keys())]],
                dtype=np.object_
            ),
            "data_names": np.array(sorted(data.keys())),
            "index": data[sorted(data.keys())[0]].obs.index.values,
        }
        for i, layer in enumerate(sorted(data.keys())):
            surer_mat_data["truelabel"][0,i] = np.array(placeholder_labels, dtype=np.uint8)
            surer_mat_data["data"][0,i] = np.array(np.array(data[layer].X.T, np.float32), dtype=np.uint8)

        spio.savemat(os.path.join(DATA_PATH,f"tcga-{cancer}-surerinput.mat"), surer_mat_data)

# %%
