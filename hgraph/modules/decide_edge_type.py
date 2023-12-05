
import torch

# helper functions, decide the type of a edge


###############################################################
# For distance graph conv
###############################################################

def decide_edge_type_distance(vec: torch.tensor, 
                        method="predefined",
                        return_edge_length=True):
    """
    classify each vec into many categories. 
    vec: N x 3
    ret: N
    """
    if method == "predefined":
        return decide_edge_type_predefined_distance(vec, return_edge_length=return_edge_length)
    else:
        raise NotImplementedError


def decide_edge_type_predefined_distance(vec: torch.tensor, epsilon=0.00001, return_edge_length=True):
    """
    classify each vec into N categories, according to the length of the vcector. 
    the last category is self-loop
    vec: N x 3
    ret: N
    """
    positive_x = torch.maximum(vec[..., 0], torch.zeros_like(vec[..., 0]))
    positive_y = torch.maximum(vec[..., 1], torch.zeros_like(vec[..., 0]))
    positive_z = torch.maximum(vec[..., 2], torch.zeros_like(vec[..., 0]))
    negative_x = - torch.minimum(vec[..., 0], torch.zeros_like(vec[..., 0]))
    negative_y = - torch.minimum(vec[..., 1], torch.zeros_like(vec[..., 0]))
    negative_z = - torch.minimum(vec[..., 2], torch.zeros_like(vec[..., 0]))

    ary = torch.stack([positive_x, positive_y, positive_z, negative_x, negative_y, negative_z])
    ary = ary.transpose(0, 1)
    
    device = vec.device
 
    edge_type = torch.ones([len(vec), 1]).to(device) * 999
    vec_length = torch.norm(ary, dim=1)
    
    # edge_length > eps          --> type 0 edge
    # edge_length <= eps         --> type 1 edge (self-loop)
    
  #  print(thres_1)
    dist_threshold = [epsilon]
    
    
    for i in range(len(dist_threshold)-1, -1, -1):
        dist_mask = vec_length > dist_threshold[i]
        edge_type[dist_mask] = i
        
    # self-loop
    self_loops_mask = vec_length <= epsilon
    edge_type[self_loops_mask] = len(dist_threshold)
    
    # squeeze to 1d tensor
    edge_type = edge_type.squeeze(-1)
    edge_type = edge_type.long()
    
    
    # assertion, suppose there are N thresholds (epsilon included), make sure that 
    # the type of different edges are at most N+1
   # breakpoint()
    assert edge_type.max() == len(dist_threshold)     # there must be edges indexed N, since self-loop exists in all meshes
    assert edge_type.min() >= 0                       # note edge_type indexed N-1 may not exists, since edges of that length may not exists in this mesh

    if return_edge_length == True:
        return edge_type, vec_length
    else:
        return edge_type

