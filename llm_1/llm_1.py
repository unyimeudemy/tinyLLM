"""
A toy autoregressive character-level model

- Character-level tokenization
- Embeddings
- One attention layer
- One feedforward layer
- Predict next character
- 1 block
- 1 attention head
- Feedforward and residual
- No stacking (no multiple blocks)
- No normalization or dropout
- But 100% functional
- model size is ___ parameters

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

training_data = """
Once upon a time, in a small village by the river, there lived a kind baker. 
Every morning he woke up early to bake bread, cakes, and sweet rolls for the people. 
The children loved the smell of fresh bread, and they often gathered by the shop to
watch him work. One day, a stray cat wandered into the bakery. The baker gave it 
some warm milk and a cozy place to sleep. From then on, the cat stayed by his side,
greeting the villagers every morning with a cheerful meow. 
"""

"""------- Tokenizer --------"""
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
context_window = 256
input_feature = []
target_label = []
encoded_training_data = encode(training_data)

if len(training_data) >= context_window + 1:
    for i in range(len(training_data) - context_window):
        input_feature.append(encoded_training_data[i : i+context_window])
        target_label.append(encoded_training_data[i+1 : i+context_window+1])
else:
    print('Data too short for given context window')

input_feature = torch.tensor(input_feature)
target_label = torch.tensor(target_label)


vocab_size = len(sorted_text_char)

"""
During inference in an autoregressive model (like GPT):
- You start with a prompt (say 50 tokens).
- The model then generates new tokens one by one.
- The prompt + generated tokens together cannot exceed max_sequence_length.
"""
max_sequence_length = 256  

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.token_embedding_matrix = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_matrix = nn.Embedding(max_sequence_length, n_embed)
        self.attention_layer = nn.MultiheadAttention(n_embed, num_heads=1, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, input_feature):
        token_embedding = self.token_embedding_matrix(input_feature)
        _, number_of_token = input_feature.shape
        positions = torch.arange(number_of_token, device=input_feature.device).unsqueeze(0)
        position_embedding = self.positional_embedding_matrix(positions)
        input_feature = token_embedding + position_embedding
        mask = torch.triu(
            torch.ones(
                (number_of_token, number_of_token),
                device=input_feature.device,
                dtype=torch.bool
            ),
            diagonal=1
        )
        attention_output, _ = self.attention_layer(
            input_feature,
            input_feature,
            input_feature,
            attn_mask=mask
        )
        input_feature = attention_output + input_feature
        feed_forward_results = self.feed_forward(input_feature)
        input_feature = feed_forward_results + input_feature  
        logits = self.lm_head(input_feature)
        return logits
    



# model = TinyTransformer(vocab_size=len(string_to_index), n_embed=20)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# def train_llm():
#     for epoch in range(2000):
#         logits = model(input_feature)
#         loss = F.cross_entropy(logits.view(-1, len(string_to_index)), target_label.view(-1))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
#     # save_model_weights(model)


# def infer(sequence: str):
#     context = torch.tensor([encode(sequence)])
#     model.eval()
#     with torch.no_grad():
#         for _ in range(50):
#             logits = model(context)
#             probs = F.softmax(logits[:, -1, :], dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)
#             context = torch.cat([context, next_token], dim=1)

#     print(decode(context[0].tolist()))
#     return decode(context[0].tolist())












model = TinyTransformer(vocab_size=len(string_to_index), n_embed=40)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def save_model_weights(model, file_path="tiny_transformer_weights.pth"):
    torch.save(model.state_dict(), file_path)
    print(f"Model weights saved to {file_path}")


def train_llm():
    for epoch in range(3000):
        logits = model(input_feature)
        loss = F.cross_entropy(logits.view(-1, len(string_to_index)), target_label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            print("=====================================================")
            print(infer_during_training("From then on, the cat stayed"))
            print("=====================================================")
    
    save_model_weights(model)


def load_model_weights(model, file_path="tiny_transformer_weights.pth"):
    """Load the model's state dictionary from a file."""
    model.load_state_dict(torch.load(file_path))
    print(f"Model weights loaded from {file_path}")
    return model

def infer(sequence: str, max_new_tokens=100):
    context = torch.tensor([encode(sequence)])
    model = TinyTransformer(vocab_size=len(string_to_index), n_embed=40)
    model = load_model_weights(model)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(context)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            context = torch.cat([context, next_token], dim=1)
    result = decode(context[0].tolist())
    print(result)
    return result


def infer_during_training(sequence: str):
    context = torch.tensor([encode(sequence)])
    model.eval()
    with torch.no_grad():
        for _ in range(100):
            logits = model(context)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            context = torch.cat([context, next_token], dim=1)

    print(decode(context[0].tolist()))
    return decode(context[0].tolist())