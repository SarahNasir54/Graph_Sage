import re
import os
import json
import jieba
from transformers import AutoTokenizer, AutoModel
import torch


file_path = 'Weibo.txt'

def get_post_details(post_id):

    json_file = os.path.join('Weibo', f"{post_id}.json")
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            post_data = json.load(f)
            if isinstance(post_data, list) and post_data:

                return post_data
            
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Tokenization (character-based for Chinese)
    text = text.replace(" ", "") 
    tokens = jieba.lcut(text)  # lcut returns a list of tokens (words)
    
    return tokens

def make_embeddings(text):
    pre_text = preprocess_text(text)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh")
    model = AutoModel.from_pretrained("BAAI/bge-large-zh")
    inputs = tokenizer(pre_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)  # CLS token

    return embedding

def extract_features(post_id):
    data = get_post_details(post_id)
    all_combined_features = []
    #pretty_json = json.dumps(data, indent=4)
    #print(pretty_json)

    for post in data:
        text = post.get("text", "")
        fcount =  post.get("followers_count", 0)
        scount = post.get("statuses_count", 0)
        friends = post.get("friends_count", 0)
        bicount = post.get("bi_followers_count", 0)
        print("text:", text)
        print("followers count:", fcount)
        print("status count:", scount)
        print("friends count:", friends)
        print("bi_followers_count", bicount)


        user_features = torch.tensor([
            fcount,
            scount,
            friends,
            bicount
            ], dtype=torch.float)
        print(user_features.shape)

        embedding = make_embeddings(text)
        print("Embedding shape:", embedding.dim())

        if embedding.dim() == 1:
             unsqueeze_embedding = embedding.unsqueeze(0)
             unsqueeze_users = user_features.repeat(unsqueeze_embedding.size(0), 1)
             combined_features = torch.cat([unsqueeze_users, unsqueeze_embedding], dim=1)
             print("Unsqueeze user shape:",unsqueeze_users.shape)
             print("Unsqueeze embedding shape: ", unsqueeze_embedding.shape)
             print("Combined features shape:", combined_features.shape)
        else:
             unsqueeze_users = user_features.repeat(embedding.size(0), 1)
             combined_features = torch.cat([unsqueeze_users, embedding], dim=1)
             print("Unsqueeze user shape:",unsqueeze_users.shape)
             print("Combined features shape:", combined_features.shape)

        all_combined_features.append(combined_features)
        print("All features:", all_combined_features)

    # Save the combined features to a PyTorch file (.pth)
    torch.save(all_combined_features, f"node_features_{post_id}.pth")

    return all_combined_features

source_posts_count = 0
# Open the file in read mode and process it line by line
with open(file_path, 'r') as file:
        for line in file:
            if source_posts_count >= 2:
                 break
            # Strip newline characters and print each line
            current_line = line.strip()

            all_ids = re.split(r'\s+', current_line)

            print("Length of all the tokens:", len(all_ids))
        
            print("label:", all_ids[1].split(":")[1])

            source_post = all_ids[2]
            print("source post:", source_post)
           
#           retweet_ids = all_ids[3:]

            node_features = extract_features(source_post)
            print(node_features)
            source_posts_count += 1

        # #print("re-tweet ids:", retweet_ids)


        # break