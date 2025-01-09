import re
from feature_extractor import get_post_details
import torch

file_path = 'Weibo.txt'

def find_edges(post_id, label):
    data = get_post_details(post_id)
    source_post = next((post for post in data if post['parent'] is None), None)

    source_id = int(source_post['id'])
    print("Source id:", source_id)
    source_mid = source_post['mid']
    print("Source mid", source_mid)
    
    # Find retweets with the same mid as the source post
    edges = [
        (source_id, int(post['id']))
        for post in data
        if post['parent'] == source_mid
    ]

    #edge_tensor = torch.tensor(edges, dtype=torch.long).t()
    #print(edge_tensor)

    # Assign sequential IDs starting from 1 to the retweets
    retweet_ids = list(range(1, len(edges) + 1))
    edges = torch.tensor([0] * len(edges), dtype=torch.long)  # Source post IDs (all 0)
    retweet_tensor = torch.tensor(retweet_ids, dtype=torch.long)  # Retweet IDs starting from 1
    
    # Stack both tensors to create the final edge tensor
    edge_tensor = torch.stack([edges, retweet_tensor])
    #print("Edge tensor:", edge_tensor.shape)
    label_tensor = torch.tensor([label], dtype=torch.long)
    #print("Label tensor:", label_tensor)


    return edge_tensor, label_tensor

# with open(file_path, 'r') as file:
#     for line in file:
#         current_line = line.strip()
#         all_ids = re.split(r'\s+', current_line)
#         source_post = all_ids[2]
#         print("source post:", source_post)

#         retweet_ids = all_ids[3:]
#         label = int(all_ids[1].split(":")[1])

#         print("label:", label)

#         edges = find_edges(source_post, label)


#         break