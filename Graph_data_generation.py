import torch
from sentiment_features import calculate_sentiment_score
import re
from edges_and_labels import find_edges
from torch_geometric.data import Data

loaded_features = torch.load(f"node_features_10031080900.pth")
loaded = torch.load(f"10031080900.pth")

def concatenate_node_features(post_id):
    all_combined_features = []
    sentiments = calculate_sentiment_score(post_id)
    for i, sentiment in enumerate(sentiments):
        #print("Sentiment features:", sentiment.shape)

        node = loaded_features[i]
        #print("Node features:", node.shape)

        concatenated_node_features = torch.cat((node, sentiment), dim=1)
        print(concatenated_node_features.shape)

        all_combined_features.append(concatenated_node_features)
        #print("All features:", all_combined_features)

    torch.save(all_combined_features, f"{post_id}.pth")

    return all_combined_features

def graph_features(post_id, label):
    data_list = []
    edges, label = find_edges(post_id, label)

    all_node_features = []
    for nodes in loaded:
        aggregated_features = nodes.mean(dim=0)  # Shape: [1797]
        all_node_features.append(aggregated_features)
    #print("x:", all_node_features[0].shape)

    x = torch.stack(all_node_features)  # dim=0 stacks the features vertically
    #print("x:", x.shape)
    edge_index = torch.tensor(edges)
    #print("Edge index shape:", edge_index.shape)
    graph_data = Data(x=x, edge_index=edges, y=label)
    torch.save(graph_data, f'Weibo_graphs/graph_data_{post_id}.pth')
    #print("Graph Data:", graph_data)

    return graph_data

# file_path = 'Weibo.txt'
# with open(file_path, 'r') as file:
#     for line in file:
#         current_line = line.strip()
#         all_ids = re.split(r'\s+', current_line)
#         source_post = all_ids[2]
#         print("source post:", source_post)

#         retweet_ids = all_ids[3:]
#         label = int(all_ids[1].split(":")[1])

#         print("label:", label)
#         graph = graph_features(source_post, label)

#         break
