# %%
import os
from itertools import chain

import gdown

import numpy as np
import scipy as sp
import scipy.io as spio
import pandas as pd

import scanpy as sc

import sklearn
import sklearn.preprocessing

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

le = sklearn.preprocessing.LabelEncoder()
labels = le.fit_transform(data["rna"].obs["cell_type"])

#%%
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
    "truelabelnames": np.array(le.classes_),
    "index": data[sorted(data.keys())[0]].obs.index.values,
}
for i, layer in enumerate(sorted(data.keys())):
    surer_mat_data["truelabel"][0,i] = np.array(labels, dtype=np.uint8)
    surer_mat_data["data"][0,i] = np.array(np.array(data[layer].X.T, np.float32), dtype=np.uint8)

spio.savemat(os.path.join(DATA_PATH,f"kotliarov2020-surerinput.mat"), surer_mat_data)

# %%
