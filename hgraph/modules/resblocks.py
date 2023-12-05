import torch
import torch.utils.checkpoint

from hgraph.hgraph import HGraph
from hgraph.modules.modules import Conv1x1BnRelu, Conv1x1, Conv1x1Bn, \
    GraphConv, GraphConvBnRelu, GraphConvBn, FcBnRelu, \
    UnpoolingGraph, PoolingGraph


class GraphResBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 bottleneck: int=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck = bottleneck
        channelb = int(out_channels / bottleneck)

        self.conv1x1a = Conv1x1BnRelu(in_channels, channelb)
        self.conv3x3 = GraphConvBnRelu(channelb, channelb)
        self.conv1x1b = Conv1x1Bn(channelb, out_channels)

        if self.in_channels != self.out_channels:
            self.conv1x1c = Conv1x1Bn(in_channels, out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, hgraph: HGraph, depth: int):
        conv1 = self.conv1x1a(data)
        conv2 = self.conv3x3(conv1, hgraph, depth)
        conv3 = self.conv1x1b(conv2)
        if self.in_channels != self.out_channels:
            data = self.conv1x1c(data)
        out = self.relu(conv3 + data)
        return out




class GraphResBlock2(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 bottleneck: int=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck = bottleneck
        channelb = int(out_channels / bottleneck)

        self.conv3x3a = GraphConvBnRelu(in_channels, channelb)
        self.conv3x3b = GraphConvBn(channelb, out_channels)

        if self.in_channels != self.out_channels:
            self.conv1x1 = Conv1x1Bn(in_channels, out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, hgraph: HGraph, depth: int):
        conv1 = self.conv3x3a(data, hgraph, depth)
        conv2 = self.conv3x3b(conv1, hgraph, depth)
        if self.in_channels != self.out_channels:
            data = self.conv1x1(data)
        out = self.relu(conv2 + data)
        return out



class GraphResBlocks(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 resblk_num, bottleneck=4,
                 resblk=GraphResBlock):
        super().__init__()
        self.resblk_num = resblk_num
        channels = [in_channels] + [out_channels] * resblk_num

        self.resblks = torch.nn.ModuleList(
            [resblk(channels[i], channels[i+1], bottleneck=bottleneck) for i in range(self.resblk_num)]
        )

    def forward(self, data: torch.Tensor, hgraph: HGraph, depth: int):
        for i in range(self.resblk_num):
            data = self.resblks[i](data, hgraph, depth)

        return data
    
    