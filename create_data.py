from torch_geometric.data import Data #, DataLoader
from torch_geometric.loader import DataLoader
import torch
import numpy as np
from torch.utils.data import Dataset, random_split#, DataLoader

import pandas as pd
from torch_geometric.data import Batch

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

class MatrixDataset(Dataset):
    def __init__(self, sour_root_dir, tar_root_dir, num_nodes=35, transform=None, pre_transform=None):
        self.sour_root_dir = sour_root_dir
        self.tar_root_dir = tar_root_dir

        self.transform = transform
        self.pre_transform = pre_transform
        self.num_nodes = num_nodes
        self.adj_matrix_size = int((num_nodes * num_nodes - num_nodes) / 2)

        self.sour_data_list = []
        self.tar_data_list = []
        self.load_data()

    def remove_rows(self, num_rows_to_remove=5):

      df1_final = self.df1.drop(self.df1.sample(num_rows_to_remove).index)
      df2_final = self.df2.drop(self.df2.sample(num_rows_to_remove).index)
      return df1_final, df2_final


    def load_data(self):
        self.df1 = pd.read_csv(self.sour_root_dir)
        self.df2 = pd.read_csv(self.tar_root_dir)
        self.df1, self.df2 = self.remove_rows()

        self.sour_data_list = self.convert_features(np.array(self.df1.iloc[:, 1:]))
        self.tar_data_list = self.convert_features(np.array(self.df2.iloc[:, 1:]))

    def convert_features(self, data):
        connection = []
        for i in range(data.shape[0]):
            feature_vector = data[i, :self.adj_matrix_size]

            # Create an upper triangular matrix with zeros on the diagonal
            upper_triangular_matrix = np.zeros((self.num_nodes, self.num_nodes))

            # Fill the upper triangular part with the features
            upper_triangular_matrix[np.triu_indices(self.num_nodes, k=1)] = feature_vector

            # Create the symmetric adjacency matrix
            adjacency_matrix = upper_triangular_matrix + upper_triangular_matrix.T

            # Normalize the adjacency matrix
            norm = (adjacency_matrix - adjacency_matrix.min()) / (adjacency_matrix.max() - adjacency_matrix.min())
            connection.append(norm)

        return connection


    def __len__(self):
        return len(self.sour_data_list)

    def __getitem__(self, idx) -> tuple:
        sour_sample_data = self.sour_data_list[idx]
        tar_sample_data = self.tar_data_list[idx]
        node_num = sour_sample_data.shape[0]

        # source_edge_weight_matrix = torch.from_numpy(sour_sample_data)
        # # Create a complete graph's edge_index
        # node_indices = torch.arange(node_num)
        # src_edge_index = torch.cartesian_prod(node_indices, node_indices).t().contiguous()
        # src_edge_attr = source_edge_weight_matrix.flatten().unsqueeze(0).t()

        # # sour_node_features = np.sum(sour_sample_data, axis=1, keepdims=True)
        # # sour_x = torch.from_numpy(sour_sample_data).float()
        # sour_node_features = np.sum(sour_sample_data, axis=1, keepdims=True)
        # # sour_x = torch.from_numpy(sour_node_features).float()
        # # sour_data = Data(x=sour_x, edge_index=src_edge_index, edge_attr=src_edge_attr)
        # sour_x = torch.from_numpy(sour_sample_data).float()
        # sour_data = Data(x=sour_x, edge_index=src_edge_index, edge_attr=src_edge_attr)

        # target_edge_weight_matrix = torch.from_numpy(tar_sample_data)
        # # Create a complete graph's edge_index
        # node_indices = torch.arange(node_num)
        # tar_edge_index = torch.cartesian_prod(node_indices, node_indices).t().contiguous()
        # tar_edge_attr = target_edge_weight_matrix.flatten().unsqueeze(0).t()
        # # tar_x = torch.from_numpy(tar_sample_data).float()
        # tar_node_features = np.sum(tar_sample_data, axis=1, keepdims=True)
        # # tar_x = torch.from_numpy(tar_node_features).float()
        # tar_x = torch.from_numpy(tar_sample_data).float()
        # tar_data = Data(x=tar_x, edge_index=tar_edge_index, edge_attr=tar_edge_attr)
        # return sour_data.to(device), tar_data.to(device)
        source_edge_weight_matrix = torch.from_numpy(sour_sample_data)
        # Create a complete graph's edge_index
        node_indices = torch.arange(node_num)
        src_edge_index = torch.cartesian_prod(node_indices, node_indices).t().contiguous()
        src_edge_attr = source_edge_weight_matrix.flatten().unsqueeze(0).t()

        # sour_x = torch.from_numpy(sour_sample_data).float()
        sour_node_features = np.sum(sour_sample_data, axis=1, keepdims=True)
        sour_x = torch.from_numpy(sour_node_features).float()
        sour_data = Data(x=sour_x, edge_index=src_edge_index, edge_attr=src_edge_attr)

        target_edge_weight_matrix = torch.from_numpy(tar_sample_data)
        # Create a complete graph's edge_index
        node_indices = torch.arange(node_num)
        tar_edge_index = torch.cartesian_prod(node_indices, node_indices).t().contiguous()
        tar_edge_attr = target_edge_weight_matrix.flatten().unsqueeze(0).t()
        # tar_x = torch.from_numpy(tar_sample_data).float()

        tar_node_features = np.sum(tar_sample_data, axis=1, keepdims=True)
        tar_x = torch.from_numpy(tar_node_features).float()
        tar_data = Data(x=tar_x, edge_index=tar_edge_index, edge_attr=tar_edge_attr)
        return sour_data.to(device), tar_data.to(device)

def custom_collate_fn(batch):
    return Batch.from_data_list(batch)

def return_dataset_individuals(batch_size=3, split_size=0.7):
    t1_file = 'data\\train_t0.csv'
    t2_file = 'data\\train_t1.csv'

    # Create datasets (assuming MatrixDataset is a custom dataset class)
    t1_dataset = MatrixDataset(sour_root_dir=t1_file, tar_root_dir=t1_file)
    t2_dataset = MatrixDataset(sour_root_dir=t2_file, tar_root_dir=t2_file)

    # Split dataset into train and test sets for both t1 and t2 datasets
    def split_dataset(dataset, split_size):
        dataset_size = len(dataset)
        train_size = int(split_size * dataset_size)
        test_size = dataset_size - train_size
        return random_split(dataset, [train_size, test_size])

    t1_train_dataset, t1_test_dataset = split_dataset(t1_dataset, split_size)
    t2_train_dataset, t2_test_dataset = split_dataset(t2_dataset, split_size)

    # Create DataLoader for training and testing
    # t1_train_loader = DataLoader(t1_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    # t1_test_loader = DataLoader(t1_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    # t2_train_loader = DataLoader(t2_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    # t2_test_loader = DataLoader(t2_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    t1_train_loader = DataLoader(t1_train_dataset, batch_size=batch_size, shuffle=True)
    t1_test_loader = DataLoader(t1_test_dataset, batch_size=batch_size, shuffle=False)
    t2_train_loader = DataLoader(t2_train_dataset, batch_size=batch_size, shuffle=True)
    t2_test_loader = DataLoader(t2_test_dataset, batch_size=batch_size, shuffle=False)

    return t1_train_loader, t1_test_loader, t2_train_loader, t2_test_loader

def return_dataset_all(batch_size=3, split_size=0.7):
    t1_file = './train_t0.csv'
    t2_file = './train_t1.csv'

    # Create datasets (assuming MatrixDataset is a custom dataset class)
    dataset = MatrixDataset(sour_root_dir=t1_file, tar_root_dir=t2_file)

    # Split dataset into train and test sets for both t1 and t2 datasets
    def split_dataset(dataset, split_size):
        dataset_size = len(dataset)
        train_size = int(split_size * dataset_size)
        test_size = dataset_size - train_size
        return random_split(dataset, [train_size, test_size])

    train_dataset, test_dataset = split_dataset(dataset, split_size)

    # Create DataLoader for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader