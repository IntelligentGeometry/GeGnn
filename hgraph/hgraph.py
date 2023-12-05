from __future__ import annotations
from typing import Callable, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import SAGEConv
#from torch_geometric.nn import avg_pool, voxel_grid

#from utils.thsolver import default_settings

from .modules.decide_edge_type import *

"""
graph neural network coarse to fine data structure
keep records of a "graph tree"
"""


class Data:
    def __init__(self, x=None, edge_index=None, edge_attr=None):
        """
        a rewrite of torch_geometric.data.Data
        get rid of its self-aleck re-indexing
        :param edge_attr: the type of edges.
        """
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def to(self, target):
        # trans data from gpu to cpu or vice versa
        self.x = self.x.to(target)
        self.edge_index = self.edge_index.to(target)
        if self.edge_attr != None:
            self.edge_attr = self.edge_attr.to(target)
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")



def avg_pool(
    cluster: torch.Tensor,
    data: Data,
    transform: Optional[Callable] = None,
) -> Data:
    """a wrapper of torch_geometric.nn.avg_pool"""
    data_torch_geometric = torch_geometric.data.Data(x=data.x, edge_index=data.edge_index)
    new_data = torch_geometric.nn.avg_pool(cluster, data_torch_geometric, transform=transform)
    ret = Data(x=new_data.x, edge_index=new_data.edge_index)
    return ret


def avg_pool_maintain_old(
    cluster: torch.Tensor,
    data: Data,
    transform: Optional[Callable] = None,
):
    """a wrapper of torch_geometric.nn.avg_pool, but maintain the old graph"""
    data_torch_geometric = torch_geometric.data.Data(x=data.x, edge_index=data.edge_index)
    new_layer = torch_geometric.nn.avg_pool(cluster, data_torch_geometric, transform=transform)
    # connect the corresponding node in the two layers
    


def pooling(data: Data, size: float, normal_aware_pooling):
    """
    do pooling according to x. a new graph (x, edges) will be generated after pooling.
    This function is a wrapper of some funcs in pytorch_geometric. It assumes the 
    object's coordinates range from -1 to 1.
    
    normal_aware_pooling: if True, only data.x[..., :3] will be used for grid pooling.
    """
    assert type(size) == float
    if normal_aware_pooling == False:
        x = data.x[..., :3]
    else:
        x = data.x[..., :6]
        # we assume x has 6 feature channels, first 3 xyz, then 3 nxnynz
        # grid size here waits for fine-tuning
        n_size = size * 3.   # a hyper parameter, controling "how important the normal vecs are, compared to xyz coords"
        size = [size, size, size, n_size, n_size, n_size]

    edges = data.edge_index
    cluster = torch_geometric.nn.voxel_grid(pos=x, size=size, batch=None,
                                            start=[-1, -1, -1.], end=[1, 1, 1.])

    # keep max index smaller than # of unique (e.g., [4,5,4,3] --> [0,1,0,2])
    mapping = cluster.unique()
    mapping += mapping.shape[0]
    cluster += mapping.shape[0]

    for i in range(int(mapping.shape[0])):  # maybe some optimization here to remove the for loop
        cluster[cluster == mapping[i]] = i

    return cluster #.contiguous()




def add_self_loop(data: Data):
    # avg_pool will clear the self loop in the graph. here we add it back
    device = data.x.device
    n_vert = data.x.shape[0]
    self_loops = torch.tensor([[i for i in range(int(n_vert))]])
    self_loops = self_loops.repeat(2, 1)
    new_edges = torch.zeros([2, data.edge_index.shape[1] + n_vert], dtype=torch.int64)
    new_edges[:, :data.edge_index.shape[1]] = data.edge_index
    new_edges[:, data.edge_index.shape[1]:] = self_loops

    return Data(x=data.x, edge_index=new_edges).to(device)




class HGraph:
    """
    HGraph stands for "Hierarchical Graph"

    notes:
    - coordinates of input vertices should be in [-1, 1]
    - this class handles xyz coordinates/normals, not input feature

    """
    def __init__(self, depth: int=5,
                 smallest_grid=2/2**5,
                 batch_size: int=1,
                 adj_layer_connected=False):
        """

        suppose depth=3:
        self.treedict[0] = original graph
        self.treedict[1] = merge verts in voxel with edge length = smallest_grid
        self.treedict[2] = merge verts in voxel with edge length = smallest_grid * (2**1) 
        self.treedict[3] = merge verts in voxel with edge length = smallest_grid * (2**2) 

        __init__ method only specify hyper parameters of a hgraph.
        The real construction of hgraph happens in build_single_hgraph() or merge_hgraph()
        """

        assert smallest_grid * (2**(depth-1)) <= 2
        assert depth >= 0

        self.device = "cuda"
        self.depth = depth
        self.batch_size = batch_size
        self.smallest_grid = smallest_grid
        self.normal_aware_pooling = True

        self.vertices_sizes = {}
        self.edges_sizes = {}
        #
        self.treedict = {}
        self.cluster = {}

    def build_single_hgraph(self, original_graph: Data):
        """
        build a graph-tree of **one** graph
        """
        assert type(original_graph) == Data
        

        graphtree = {}
        cluster = {}
        vertices_size = {}
        edges_size = {}

        for i in range(self.depth+1):
            if i == 0:
                original_graph = add_self_loop(original_graph)
                # if original graph do not have edge types, assign it 
                if original_graph.edge_attr == None:
                    edges = original_graph.x[original_graph.edge_index[0]] \
                            - original_graph.x[original_graph.edge_index[1]]
                    edges_attr = decide_edge_type_distance(edges, return_edge_length=False)
                    original_graph.edge_attr = edges_attr
                graphtree[0] = original_graph
                cluster[0] = None
                edges_size[0] = original_graph.edge_index.shape[1]
                vertices_size[0] = original_graph.x.shape[0]
                continue

            clst = pooling(graphtree[i-1], self.smallest_grid * (2**(i-1)), normal_aware_pooling=self.normal_aware_pooling)
            new_graph = avg_pool(cluster=clst, data=graphtree[i-1], transform=None)
            new_graph = add_self_loop(new_graph)
            # assign edge type
            edges = new_graph.x[new_graph.edge_index[0]] \
                    - new_graph.x[new_graph.edge_index[1]]
            edges_attr = decide_edge_type_distance(edges, return_edge_length=False)
            new_graph.edge_attr = edges_attr

            graphtree[i] = new_graph
            cluster[i] = clst
            edges_size[i] = new_graph.edge_index.shape[1]
            vertices_size[i] = new_graph.x.shape[0]

        self.treedict = graphtree
        self.cluster = cluster
        self.vertices_sizes = vertices_size
        self.edges_sizes = edges_size

        #self.export_obj()


    # @staticmethod
    def merge_hgraph(self, original_graphs: list[HGraph], debug_report=False):
        """
        merge multi hgraph into a large hgraph

        """
        assert len(self.cluster) == 0 and len(self.treedict) == 0, "please call this function on a new instance"
        assert original_graphs.__len__() == self.batch_size, "please make sure the batch size is correct"

        # re-indexing
        for d in range(self.depth+1):
            # merge vertices for every layer
            num_vertices = [0]
            for i, each in enumerate(original_graphs):
                num_vertices.append(each.vertices_sizes[d])
            cum_sum = torch.cumsum(torch.tensor(num_vertices), dim=0)
            for i in range(original_graphs.__len__()):
                original_graphs[i].treedict[d].edge_index += cum_sum[i]
                # cluster is None at d=0
                if d != 0:
                    original_graphs[i].cluster[d] += cum_sum[i]

        # merge 
        for d in range(self.depth+1):
            graphtrees_x, graphtrees_e, graphtrees_e_type, clusters = [], [], [], []
            for i in range(original_graphs.__len__()):
                graphtrees_x.append(original_graphs[i].treedict[d].x)
                graphtrees_e.append(original_graphs[i].treedict[d].edge_index)
                graphtrees_e_type.append(original_graphs[i].treedict[d].edge_attr)
                clusters.append(original_graphs[i].cluster[d])
            # construct new graph
            temp_data = Data(x=torch.cat(graphtrees_x, dim=0),
                             edge_index=torch.cat(graphtrees_e, dim=1),
                             edge_attr=torch.cat(graphtrees_e_type, dim=0)   # edge_attr shape: [E]
                             )
            # construct new cluster
            if d != 0:
                temp_clst = torch.cat(clusters, dim=0)
            else:
                temp_clst = None
            self.treedict[d] = temp_data
            self.cluster[d] = temp_clst
            self.edges_sizes = temp_data.edge_index.shape[1]
            self.vertices_sizes = len(temp_data.x)

        # sanity check
        if debug_report == True:
            # a simple unit test
            for d in range(self.depth+1):
                num_edges_before = 0
                for i in range(original_graphs.__len__()):
                    num_edges_before += original_graphs[i].treedict[d].edge_index.shape[1]
                num_edges_after = self.treedict[d].edge_index.shape[1]
                print(f"Before merge, at d={d} there's {num_edges_before} edges; {num_edges_after} afterwards")




    #####################################################
    # Util
    #####################################################

    def cuda(self):
        # move all tensors to cuda
        for each in self.treedict.keys():
            self.treedict[each] = self.treedict[each].cuda()
        for each in self.cluster.keys():
            if self.cluster[each] is None:
                continue
            self.cluster[each] = self.cluster[each].cuda()
        return self


    