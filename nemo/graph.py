import warnings

from typing import Union

import numpy as np
import scipy.sparse as spsparse
import networkx as nx

try:
    import igraph as ig

    def get_igraph_from_networkx(G:nx.Graph) -> ig.Graph:
        raise NotImplementedError()

    def get_igraph_from_scipy_sparse(adj:spsparse.sparray, directed=False, weighted=True) -> ig.Graph:
        srcs, tgts = adj.nonzero()
        w = adj[srcs, tgts]
        if isinstance(w, np.matrix):
            w = w.A.flatten()
        ig_rna = ig.Graph(directed=directed)
        ig_rna.add_vertices(adj.shape[0])
        ig_rna.add_edges(list(zip(srcs,tgts)))
        if weighted:
            ig_rna.es["weight"] = w
        return ig_rna
except ImportError:
    warnings.warn("IGraph is not installed, some functionality may not be present!")

try:
    import leidenalg
    def get_leiden(g:Union[nx.Graph,np.ndarray,spsparse.sparray], resolution:float=1, partition_type=None):
        if partition_type is None:
            partition_type = leidenalg.RBConfigurationVertexPartition
        if isinstance(g,nx.Graph):
            g_ig = get_igraph_from_networkx(g)
        elif isinstance(g,(np.ndarray,spsparse.sparray,spsparse.spmatrix)):
            g_ig = get_igraph_from_scipy_sparse(g)
        else:
            g_ig = g
        part = leidenalg.find_partition(g_ig, partition_type, resolution_parameter = resolution)
        return part
except ImportError:
    warnings.warn("leidenalg is not installed, some functionality may not be present!")

def get_louvain(g:np.ndarray, resolution:float=1):
    communities = nx.community.louvain_communities(g, resolution=resolution)
    clustering = np.empty([len(g.nodes)])
    for cluster in range(len(communities)):
        #18m59.9s
        #clustering[list(communities[cluster])] = cluster 
        #18m6.3s
        #for node in communities[cluster]: 
        #    clustering[node] = cluster
        #16m11.4s
        np.put(clustering, list(communities[cluster]), cluster)
    return communities, clustering