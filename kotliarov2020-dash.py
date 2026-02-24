# %%
from functools import partial
import os

import gdown

import numpy as np
import pandas as pd

import scanpy as sc

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sklearn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold, train_test_split, cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans
import sklearn.metrics

import networkx as nx
import igraph
import sklearn.model_selection

from nemo.models.subsetcontrastive import SubsetContrastive
from nemo.graph import get_leiden

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
sc.pl.umap(data["rna"], color=['batch', 'cell_type', 'cluster_level2', 'cluster_level3'], ncols=1)

#%%
n_clust = 50
roc_ovr = partial(sklearn.metrics.roc_auc_score,multi_class="ovr")
roc_ovr.__name__ = sklearn.metrics.roc_auc_score.__name__
scorer = sklearn.metrics.make_scorer(roc_ovr, needs_proba=True)
#%%
xtr, xte, ctr, cte = train_test_split(data["rna"].X, data["rna"].obs["cell_type"])
xall = data["rna"].X
#%%
km = KMeans(n_clust)
yclusttr = km.fit_predict(xtr)
yclustte = km.predict(xte)
yclustall = km.predict(data["rna"].X)
#%%
lr = LogisticRegression()
lr.fit(xtr, yclusttr)
#%%
scoretr = lr.score(xtr,yclusttr)
scorete = lr.score(xte,yclustte)
scoreall = lr.score(xall,yclustall)
#%%
scorescv = []
for folditr, foldite in StratifiedKFold(3).split(xtr, yclusttr):
    print(folditr.shape, foldite.shape)
    foldlr = LogisticRegression()
    foldlr.fit(xtr[folditr], yclusttr[folditr])
    scorescv.append(scorer(foldlr, xtr[foldite], yclusttr[foldite]))
np.mean(scorescv)
#%%
print(scoretr, scorete, scoreall)
#%%
ylrte = lr.predict(xte)
ylrtr = lr.predict(xtr)
ylrall = lr.predict(data["rna"].X)
#%%
def get_self_projection_confusion_matrix(y_true, y_pred, normalize=None, normalize_after_removing_eye=True, eye_to_nan=False):
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    eye = np.eye(cm.shape[0], dtype=bool)
    if normalize_after_removing_eye:
        cm_eye = cm[eye]
        cm[eye] = 0
    match normalize:
        case None:
            pass
        case "pred":
            cm = cm/np.sum(cm,0,keepdims=True)
        case "true":
            cm = cm/np.sum(cm,1,keepdims=True)
        case "all":
            cm = cm/np.sum(cm,keepdims=True)
        case _:
            raise ValueError(f"normalize must be either None, 'pred', 'true' or 'all', but it is {normalize}")
    if not normalize_after_removing_eye:
        cm_eye = cm[eye]
        cm[eye] = 0
    cm[np.isnan(cm)] = 0.
    if eye_to_nan:
        cm[eye] = np.nan
    #if symmetric:
    #    cm = (cm+cm.T)/2
    return cm, cm_eye

# %%
cm, cmeye = get_self_projection_confusion_matrix(yclustall, ylrall, normalize="true")
plt.imshow(cm)
# %%
cm, cmeye = get_self_projection_confusion_matrix(yclusttr, ylrtr, normalize="true")
plt.imshow(cm)
# %%
cm, cmeye = get_self_projection_confusion_matrix(yclustte, ylrte, normalize="true", normalize_after_removing_eye=False)
#cm = (cm+cm.T)/2
cm = np.where(cm<cm.T,cm.T,cm)
plt.imshow(cm)
plt.show()
plt.close()
if False:
    DESIRED_CONNECTED_COMPONENTS = 10
    thresh = 0
    old_ncc = None
    while True:
        nodes = list(range(cm.shape[0]))
        edges = np.stack((cm>thresh).nonzero()).T.tolist()
        print(thresh)
        nxg = nx.Graph()
        nxg.add_nodes_from(nodes)
        nxg.add_edges_from(edges)
        igg =  igraph.Graph(n=len(nodes), edges=edges)
        commu = get_leiden(igg)
        nx.draw(nxg, node_color=commu._membership, node_size=20)
        plt.show()
        plt.close()
        ncc = nx.number_connected_components(nxg)
        if old_ncc is not None and old_ncc==ncc:
            break
        if ncc<DESIRED_CONNECTED_COMPONENTS:
            thresh = (thresh + np.max(cm))/2
        elif ncc>DESIRED_CONNECTED_COMPONENTS:
            thresh = (thresh)/2
#%%
thresh = np.max(cm)-0.01
plt.imshow(cm>thresh)
nodes = list(range(cm.shape[0]))
edges = np.stack((cm>thresh).nonzero()).T.tolist()

#%%
nxg = nx.Graph()
nxg.add_nodes_from(nodes)
nxg.add_edges_from(edges)
igg = igraph.Graph(n=cm.shape[0], edges=edges)
#%%
commu = get_leiden(igg)
#%%
pos = np.stack([nx.drawing.kamada_kawai_layout(nxg)[k] for k in range(cm.shape[0])])
# %%
ax = plt.gca()
communities = np.unique(commu._membership)
cmap = plt.colormaps.get_cmap("tab20")
for s,t in nxg.edges:
    ax.add_artist(Line2D(pos[[s,t],0],pos[[s,t],1], color=(0.,0,0,0.1)))
plt.scatter(pos[:,0], pos[:,1], c=cmap(commu._membership), s=10)
plt.legend(cmap(communities),communities)

#%%
nx.draw(nxg, pos, node_color=commu._membership, node_size=20)
# %%
for c in communities:
    ys = np.where(commu._membership==c)[0]
    cell_types = []
    for y in ys:
        cell_types.extend(data["rna"].obs["cell_type"][yclustall==y])
    cell_types = pd.Series(cell_types)
    dc = cell_types.describe()
    topprop = dc["freq"]/dc["count"]
    if topprop<0.8:
        print(c, dc["top"], topprop)

# %%
