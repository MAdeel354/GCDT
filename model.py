import torch
import torch.nn as nn
from torch_geometric.nn import NNConv
from torch_geometric.nn import TransformerConv
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.solvers.diophantine.diophantine import reconstruct
from torch_geometric.nn import GCNConv, NNConv, BatchNorm
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout, Module
from torch_geometric.nn import Sequential as PygSequential

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch_geometric.nn import NNConv
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch.nn import Linear, Sequential, ReLU
from torch.nn.parameter import Parameter
from torch import mm as mm
from torch.nn import Tanh




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # First NNConv layer with edge feature processing
        lin1 = nn.Sequential(nn.Linear(1, 35), nn.ReLU())  # Transform edge features from 1 to 1225
        self.conv1 = NNConv(1, 35, lin1, aggr='mean', root_weight=True, bias=True)
        self.conv1_bn = nn.BatchNorm1d(35, eps=1e-03, momentum=0.1, affine=True)

        # Second NNConv layer
        lin2 = nn.Sequential(nn.Linear(1, 35), nn.ReLU())
        self.conv2 = NNConv(35, 1, lin2, aggr='mean', root_weight=True, bias=True)
        self.conv2_bn = nn.BatchNorm1d(1, eps=1e-03, momentum=0.1, affine=True)

        # Third NNConv layer with skip connection
        lin3 = nn.Sequential(nn.Linear(1, 35), nn.ReLU())
        self.conv3 = NNConv(1, 35, lin3, aggr='mean', root_weight=True, bias=True)
        self.conv3_bn = nn.BatchNorm1d(35, eps=1e-03, momentum=0.1, affine=True)

        # A more complex fully connected layer for edge feature processing
        self.mlp_edge = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Output layer (X with skip connections)
        self.output_layer = nn.Linear(70, 35)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.float()
        edge_attr = edge_attr.float()

        # Pass through the first NNConv layer with BatchNorm
        x1 = torch.sigmoid(self.conv1_bn(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, p=0.4, training=self.training)

        # Apply a symmetry correction (X + X.T)
        x1 = (x1 + x1.T) / 2.0
        x1.fill_diagonal_(0)

        # Pass through the second NNConv layer with BatchNorm
        x2 = torch.sigmoid(self.conv2_bn(self.conv2(x1, edge_index, edge_attr)))
        x2 = F.dropout(x2, p=0.2, training=self.training)

        # Pass through the third NNConv layer with skip connection and BatchNorm
        x3 = torch.cat([torch.sigmoid(self.conv3_bn(self.conv3(x2, edge_index, edge_attr))), x1], dim=1)

        # Split the output and combine them
        x4 = x3[:, :35]
        x5 = x3[:, 35:]

        # Combine and apply symmetry correction again
        x6 = (x4 + x5) / 2
        x6 = (x6 + x6.T) / 2.0
        x6.fill_diagonal_(0)

        # Additional MLP layer to refine the edge features
        edge_feats = self.mlp_edge(edge_attr)
        edge_features = edge_feats.view(35, 35)

        # Output layer to refine the final prediction
        x_out = torch.cat([x6, edge_features], dim=1)
        x_out = self.output_layer(x_out)

        return x_out


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = GCNConv(1, 35, cached=True)
        self.conv2 = GCNConv(35, 1, cached=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # x = torch.squeeze(x)
        x1 = F.sigmoid(self.conv1(x, edge_index))
        x1 = F.dropout(x1, training=self.training)
        x2 = F.sigmoid(self.conv2(x1, edge_index))
        #         # x2 = F.dropout(x2, training=self.training)
        return x2