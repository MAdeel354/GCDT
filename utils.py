import torch
from torch_geometric.data import Data #, DataLoader



def convert_generated_to_graph(data):

    node_indices = torch.arange(35)
    edge_index = torch.cartesian_prod(node_indices, node_indices).t().contiguous()
    edge_attr = data.flatten().unsqueeze(0).t()
    x = data.float()  # Use the tensor as is
    x = torch.sum(data, axis=1).view(-1,1)

    g_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return g_data

def frobenius_distance_loss(adj_true, adj_pred):
    return torch.sqrt(torch.sum((adj_true - adj_pred) ** 2))