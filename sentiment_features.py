import re
from transformers import AutoTokenizer, AutoModel
import torch
from snownlp import SnowNLP
from feature_extractor import get_post_details
from feature_extractor import preprocess_text

file_path = 'Weibo.txt'
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm")
model = AutoModel.from_pretrained("hfl/chinese-bert-wwm")


def get_emotional_embeddings(text):
    pre_text = preprocess_text(text)
    inputs = tokenizer(pre_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Get average of last hidden state (embedding)
    return embeddings

def calculate_sentiment_score(post_id):
    data = get_post_details(post_id)

    all_sentiment_features = []
    for post in data:
        text = post.get("text", "")
        s = SnowNLP(text)
        sentiment_score = s.sentiments  # Returns sentiment polarity score between 0 and 1
        #print("Sentiment score for post:", sentiment_score)

        embedding = get_emotional_embeddings(text)
        #print("Embedding shape:", embedding.shape)

        sentiment_tensor = torch.tensor([sentiment_score], dtype=torch.float).unsqueeze(0)
        sentiment_score_tensor = sentiment_tensor.repeat(embedding.size(0), 1)
        #print("Sentiment tensor shape:", sentiment_score_tensor.shape)

        # Concatenate sentiment score with its corresponding embedding
        combined_feature = torch.cat([sentiment_score_tensor, embedding], dim=1)  # Shape: [1, 1 + embedding_dim]
        #print("Sentiment feature shape:", combined_feature.shape)
        all_sentiment_features.append(combined_feature)

    return all_sentiment_features

# with open(file_path, 'r') as file:
#     for line in file:
#         current_line = line.strip()
#         all_ids = re.split(r'\s+', current_line)
#         source_post = all_ids[2]
#         print("source post:", source_post)

#         retweet_ids = all_ids[3:]

#         emotions = calculate_sentiment_score(source_post)
#         #print(emotions)

#         break
