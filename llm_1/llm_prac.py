# import torch
# import torch.nn as nn

# training_data = """
# Once upon a time, in a small village by the river, there lived a kind baker. 
# Every morning he woke up early to bake bread, cakes, and sweet rolls for the people. 
# The children loved the smell of fresh bread, and they often gathered by the shop to watch him work. 
# One day, a stray cat wandered into the bakery. The baker gave it some warm milk and a cozy place to sleep. 
# From then on, the cat stayed by his side, greeting the villagers every morning with a cheerful meow. 
# """


# unique_chars = set(training_data)
# sorted_unique_chars = sorted(unique_chars)

# char_to_index_map = {}
# for index, char in enumerate(sorted_unique_chars):
#     char_to_index_map[char] = index

# index_to_char_map = {}
# for char, index in char_to_index_map.items():
#     index_to_char_map[index] = char

# def encode(query: str):
#     result = []
#     for char in query:
#         result.append(char_to_index_map[char])
#     return result

# def decode(index_list):
#     result = ""
#     for index in index_list:
#         result = result + index_to_char_map[index]
#     return result

# context_window = 128
# feature_list = []
# target_list = []
# tokenized_training_data = encode(training_data)

# if len(training_data) >= context_window + 1:
#     for index in range(len(training_data) - context_window):
#         feature_list.append(tokenized_training_data[index : index+context_window])
#         target_list.append(tokenized_training_data[index+1 : index+context_window+1])
# else:
#     print("Training data is too small for context window")


# feature_tensor = torch.tensor(feature_list)
# target_tensor = torch.tensor(target_list)

# vocab_size = len(sorted_unique_chars)


# class TinyTransformer(nn.Module):
#     def __init__(self, n_embed):
#         super.__init__()
#         self.token_embedding_matrix = nn.Embedding(vocab_size, n_embed)
#         self.position_embedding_matrix = nn.Embedding(context_window, n_embed)
#         self.attention_layer = nn.MultiheadAttention(n_embed, num_heads=1, )
#         self.feed_forward = nn.Sequential(
#             nn.Linear(n_embed, 4*n_embed),
#             nn.ReLU(),
#             nn.Linear(4*n_embed, n_embed)
#         )
#         self.lm_head = nn.Linear(n_embed, vocab_size)

#     def forward(self, feature_tensor):
#         token_embedding = self.token_embedding_matrix(feature_tensor)
#         _ ,number_of_tokens = feature_tensor.shape
#         positions = torch.arange(
#             number_of_tokens,
#             device=feature_tensor.device
#         ).unsqueeze(0)
#         position_embedding = self.position_embedding_matrix(positions)
#         feature_tensor = token_embedding + position_embedding
#         mask = torch.triu(
#             torch.ones(
#                 (number_of_tokens, number_of_tokens),
#                 device=feature_tensor.device,
#                 dtype=torch.bool
#             ),
#             diagonal=1
#         )
#         attention_result = self.attention_layer(
#             feature_tensor,
#             feature_tensor,
#             feature_tensor,
#             attn_mask=mask
#         )
#         feature_tensor = feature_tensor + attention_result
#         feed_forward_result = self.feed_forward(feature_tensor)
#         feature_tensor = feature_tensor + feed_forward_result
#         logits = self.lm_head(feature_tensor)
#         return logits
    








 
