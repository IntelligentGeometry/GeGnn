import torch
import utils
from utils import ocnn

import numpy as np
from utils.thsolver import Dataset


from hgraph.hgraph import Data
from hgraph.hgraph import HGraph


class Transform(utils.ocnn.dataset.Transform):

  def __call__(self, sample: dict, idx: int):
    vertices = torch.from_numpy(sample['vertices'].astype(np.float32))
    normals = torch.from_numpy(sample['normals'].astype(np.float32))
    edges = torch.from_numpy(sample['edges'].astype(np.float32)).t().contiguous().long()
    dist_idx = sample['dist_idx'].astype(np.float32)
    dist_val = sample['dist_val'].astype(np.float32)
   # breakpoint()
    dist = np.concatenate([dist_idx, dist_val], -1)
    dist = torch.from_numpy(dist)


    rnd_idx = torch.randint(low=0, high=dist.shape[0], size=(100000,))
    dist = dist[rnd_idx]


    # normalize
    norm2 = torch.sqrt(torch.sum(normals ** 2, dim=1, keepdim=True))
    normals = normals / torch.clamp(norm2, min=1.0e-12)

    # construct hierarchical graph
    h_graph = HGraph()

    h_graph.build_single_hgraph(Data(x=torch.cat([vertices, normals], dim=1), edge_index=edges))

    return {'hgraph': h_graph,
            'vertices': vertices, 'normals': normals,
            'dist': dist, 'edges': edges}


def collate_batch(batch: list):
  # batch: list of single samples.
  # each sample is a dict with keys:
  #         edges, vertices, normals, dist

  # output: a big sample
  assert type(batch) == list

  # merge many hgraphs into one super hgraph


  outputs = {}
  for key in batch[0].keys():
    outputs[key] = [b[key] for b in batch]

  pts_num = torch.tensor([pts.shape[0] for pts in outputs['vertices']])
  cum_sum = utils.ocnn.utils.cumsum(pts_num, dim=0, exclusive=True)
  for i, dist in enumerate(outputs['dist']):
    dist[:, :2] += cum_sum[i]
  #for i, edge in enumerate(outputs['edges']):
  #  edge += cum_sum[i]
  outputs['dist'] = torch.cat(outputs['dist'], dim=0)
  

  # input feature 
  vertices = torch.cat(outputs['vertices'], dim=0)
  normals = torch.cat(outputs['normals'], dim=0)
  feature = torch.cat([vertices, normals], dim=1)
  outputs['feature'] = feature

  # merge a batch of hgraphs into one super hgraph
  hgraph_super = HGraph(batch_size=len(batch))
  hgraph_super.merge_hgraph(outputs['hgraph'])
  outputs['hgraph'] = hgraph_super

  #if (outputs['dist'].max() >= len(vertices)):
  #  print("!!!!!!!")


  # Merge a batch of octrees into one super octree
  #octree = ocnn.octree.merge_octrees(outputs['octree'])
  #octree.construct_all_neigh()
  #outputs['octree'] = octree

  # Merge a batch of points
  #outputs['points'] = ocnn.octree.merge_points(outputs['points'])
  return outputs


def get_dataset(flags):
  transform = Transform(**flags)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=np.load, take=flags.take)
  return dataset, collate_batch
