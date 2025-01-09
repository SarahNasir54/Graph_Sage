import torch 

loaded_features = torch.load(f"Weibo_graphs\graph_data_10031080900.pth")
print(loaded_features.y)