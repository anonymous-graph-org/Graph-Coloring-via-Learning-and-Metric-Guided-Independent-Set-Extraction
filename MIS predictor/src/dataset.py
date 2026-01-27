import torch
from torch.utils.data import Dataset
import networkx as nx

class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        """
        Initializes the dataset.

        Args:
            graphs (list): List of NetworkX graphs.
            labels (list): List of List of label vectors corresponding to the graphs.
        """
        self.graphs = graphs
        self.labels = labels
        

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        all_solutions = self.labels[idx]
        solutions_tensor = torch.tensor(all_solutions, dtype=torch.float32)  # Shape: [num_solutions, num_nodes]
        adjacency_matrix = torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.float32)
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        return adjacency_matrix, degree_matrix, solutions_tensor
