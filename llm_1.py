"""
A toy autoregressive character-level model

- Character-level tokenization
- Embeddings
- One attention layer
- One feedforward layer
- No masking
- Predict next character
- 1 block
- 1 attention head
- Feedforward and residual
- No stacking (no multiple blocks)
- No normalization or dropout
- But 100% functional

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

text = "The sun is"
training_data = "The sun is bright. The sky is blue. The cat sat on the mat."


"""------- Tokernizer --------"""
text_set = set(training_data)
sorted_text_char = sorted(list(text_set))


string_to_index = {}
for index, char in enumerate(sorted_text_char):
    string_to_index[char] = index

index_to_string = {}
for char, index in string_to_index.items():
    index_to_string[index] = char

def encode(string):
    encoded = []
    for char in string:
        encoded.append(string_to_index[char])
    return encoded

def decode(list_of_index):
    decoded = ''
    for index in list_of_index:
        decoded = decoded + index_to_string[index]
    return decoded



"""  
    Training data chunking for next-token prediction or
    Causal language modeling preprocessing or 
    Autoregressive training dataset construction
 """
context_window = 8
input_feature = []
target_label = []
encoded_training_data = encode(training_data)

if len(training_data) >= context_window + 1:
    for i in range(len(training_data) - context_window):
        input_feature.append(encoded_training_data[i : i+context_window])
        target_label.append(encoded_training_data[i+1 : i+context_window+1])
else:
    print('Data too long for given context window')

input_feature = torch.tensor(input_feature)
target_label = torch.tensor(target_label)

""" the total number of unique token(characters) in the training data set"""
vocab_size = len(sorted_text_char)

max_sequence_length = 128 
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.token_embedding_matrix = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_matrix = nn.Embedding(max_sequence_length, n_embed)
        self.attention_layer = nn.MultiheadAttention(n_embed, num_heads=1, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, vocab_size)
        )
    
    def forward(self, input_feature):
        number_of_batch, number_of_token = input_feature.shape
        token_embedding = self.token_embedding_matrix(input_feature)
        positions = torch.arange(number_of_token).unsqueeze(0)
        position_embedding = self.positional_embedding_matrix(positions)
        input_feature = token_embedding + position_embedding
        attention_output, _ = self.attention_layer(
            input_feature,
            input_feature,
            input_feature
        )
        input_feature = attention_output + input_feature
        logits = self.feed_forward(input_feature)
        return logits
    

model = TinyTransformer(vocab_size=len(string_to_index), n_embed=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3000):
    logits = model(input_feature)
    loss = F.cross_entropy(logits.view(-1, len(string_to_index)), target_label.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if epoch % 50 == 0:
    #     print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


context = torch.tensor([encode("The cat sat ")])
model.eval()
with torch.no_grad():
    for _ in range(50):
        logits = model(context)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)

print(decode(context[0].tolist()))
