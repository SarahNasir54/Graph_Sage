import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import DataLoader
from torch.nn import ModuleList
import re
from Graph_data_generation import graph_features

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.convs = ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)  # Apply non-linearity
            x = F.dropout(x, p=0.5, training=self.training)  # Dropout for regularization
        x = self.convs[-1](x, edge_index)  # Final layer
        return x
    
file_path = 'Weibo.txt'
with open(file_path, 'r') as file:
    for line in file:
        current_line = line.strip()
        all_ids = re.split(r'\s+', current_line)
        source_post = all_ids[2]
        print("source post:", source_post)

        retweet_ids = all_ids[3:]
        label = int(all_ids[1].split(":")[1])

        print("label:", label)
        graph_data = graph_features(source_post, label)
        print(graph_data.x.shape)
        print(graph_data.edge_index.shape)
        model = GraphSAGE(input_dim=1797, hidden_dim=128, output_dim=2)  # Example: binary classification
        model.eval()

        output = model(graph_data.x, graph_data.edge_index)

        print("Output shape:", output.shape)

        break
    

