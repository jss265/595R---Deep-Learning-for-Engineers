import torch
import torch as nn
from torch_geometric.nn import MessagePassing

class GraphNetwwork(MessagePassing):
    def __init__(sellf, aggr='add'):
        super().__init__(aggr=aggr)
    
    def forward(self, x, edge_index):
        # x is [num nodes x incannels]
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        xcat = torch.cat([x_i, x_j], dim=1)
        message = NN1(xcat)
        # returns a force

    def update(self, aggr_out, x):
        xcat = torch.cat([aggr_out, x], dim=1)
        update = NN2(xcat)
        # returns an acceleration
