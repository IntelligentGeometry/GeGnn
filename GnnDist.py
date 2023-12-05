import torch
from utils.thsolver import default_settings 
# initialize global settings
default_settings._init()
from utils.thsolver.config import parse_args
FLAGS = parse_args()
default_settings.set_global_values(FLAGS)


from utils import thsolver
import hgraph

from dataset_ps import get_dataset



def get_parameter_number(model):
    """print the number of parameters in a model on terminal """
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_num}, trainable: {trainable_num}")
    return {'Total': total_num, 'Trainable': trainable_num}


class GnnDistSolver(thsolver.Solver):

  def get_model(self, flags):
   
    if flags.name.lower() == 'unet':
      model = hgraph.models.graph_unet.GraphUNet(
          flags.channel, flags.nout, flags.interp, flags.nempty)
    
      
   #   model = ocnn.models.octree_unet.OctreeUnet(flags.channel, flags.nout, flags.interp, flags.nempty)
    else:
      raise ValueError
    
  
    # overall num parameters
    get_parameter_number(model)
    return model

  def get_dataset(self, flags):
    return get_dataset(flags)


  def model_forward(self, batch):
    """equivalent to `self.get_embd` + `self.embd_decoder_func` """

    data = batch["feature"].cuda()
    hgraph = batch['hgraph'].cuda()
    dist = batch['dist'].cuda()

    pred = self.model(data, hgraph, hgraph.depth, dist)
    return pred
  
  def get_embd(self, batch):
    """only used in visualization!"""
    data = batch["feature"].cuda()
    hgraph = batch['hgraph'].cuda()
    dist = batch['dist'].cuda()
     
    embedding = self.model(data, hgraph, hgraph.depth, dist, only_embd=True)
    return embedding
  
  def embd_decoder_func(self, i, j, embedding):
    """only used in visualization!"""
    i = i.long()
    j = j.long()
    embd_i = embedding[i].squeeze(-1)
    embd_j = embedding[j].squeeze(-1)
    embd = (embd_i - embd_j) ** 2
    pred = self.model.embedding_decoder_mlp(embd)
    pred = pred.squeeze(-1)
    return pred

  def train_step(self, batch):
    pred = self.model_forward(batch)
    loss = self.loss_function(batch, pred)
    return {'train/loss': loss}
  
  def test_step(self, batch):
    pred = self.model_forward(batch)
    loss = self.loss_function(batch, pred)
    return {'test/loss': loss}

  def loss_function(self, batch, pred):
    dist = batch['dist'].cuda()
    gt = dist[:, 2]
    
    # there are many kind of losses that may apply:

    # option 1: Mean Absolute Error, MAE
    #loss = torch.abs(pred - gt).mean()

    # option 2: relative MAE
    loss = (torch.abs(pred - gt) / (gt + 1e-3)).mean()

    # option 3: Mean Squared Error, MSE
    #loss = torch.square(pred - gt).mean()
    
    # option 4: relative MSE
    #loss = torch.square((pred - gt) / (gt + 1e-3)).mean()

    # option 5: root mean squared error, RMSE
    #loss = torch.sqrt(torch.square(pred - gt).mean())
    
    # clamp
    loss = torch.clamp(loss, -10, 10)

    return loss


#def visualization(ret):
    # open the render to visualize the result
 #   print("Establishing WebSocket and Visualization!")
  #  asyncio.run(interactive.main(ret))



if __name__ == "__main__":
    ret = GnnDistSolver.main()
    #visualization(ret)

