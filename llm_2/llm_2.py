import torch
import torch.nn as nn
import torch.nn.functional as F
import math

training_data = """
In a quiet valley surrounded by soft, rolling hills, there was a bustling town known for its colorful markets and friendly people. 
Every sunrise, merchants filled the streets with fruits, spices, fabrics, and trinkets from distant lands. 
Children hurried across cobblestone paths on their way to lessons, waving to neighbors and laughing in the crisp morning air. 
At the edge of the town square stood a tall clock tower, its bells ringing proudly to welcome each new day. 

In this town lived a humble shoemaker named Elias. He worked in a tiny workshop just beside the clock tower, 
crafting sturdy boots and elegant slippers for anyone who came through his door. 
Elias loved his craft deeply, and though he did not have much, he always offered a warm smile and a kind word. 
He enjoyed listening to stories from travelers, especially those who spoke of oceans, deserts, and kingdoms beyond the horizon. 

One rainy afternoon, a mysterious traveler arrived, soaked from head to toe. 
Without hesitation, Elias invited the stranger in, offering him a seat by the fire and a steaming cup of tea. 
The traveler, grateful for the shoemaker’s kindness, shared tales of great ships and glittering cities lit by lanterns of silver. 
Elias listened with shining eyes, feeling his heart fill with both wonder and longing. 

Days passed, and the traveler remained in the town, helping Elias in the workshop while his boots were being repaired. 
Together, they crafted shoes for children, dancers, farmers, and soldiers. 
Their laughter spilled out onto the streets, and soon the townspeople looked forward to the joyful sound that came from the workshop each day. 
The traveler taught Elias songs from faraway lands, and Elias shared with him the comfort of homemade soup and freshly baked bread. 

One bright morning, the bells of the clock tower rang louder than usual, echoing across the valley. 
The traveler announced that it was time for him to continue his journey. 
Elias felt a heaviness in his heart, but he gifted the traveler a pair of finely crafted boots, 
stitched with care and lined with soft wool to protect his feet on cold nights. 
In return, the traveler gave Elias a small silver compass, promising that their paths would someday cross again. 

From that day forward, Elias continued his work with renewed spirit. 
He sang the songs he had learned and welcomed every visitor with warmth and curiosity. 
Though he stayed in his little shop by the clock tower, the compass reminded him that the world was vast and full of possibility. 
And every evening, as the sun dipped behind the hills, Elias would step outside, gaze at the horizon, and dream of adventures yet to come.

"""

unique_chars = set(training_data)
sorted_unique_char = sorted(unique_chars)

char_to_index_map = {}
for index, char in enumerate(sorted_unique_char):
    char_to_index_map[char] = index

index_to_char_map = {}
for char, index in char_to_index_map.items():
    index_to_char_map[index] = char

def encode(query):
    result = []
    for char in query:
        result.append(char_to_index_map[char])
    return result

def decode(index_list):
    result = ""
    for index in index_list:
        result = result + index_to_char_map[index]
    return result

context_window = 128
feature_list = []
target_list = []
tokenized_training_data = encode(training_data)

if len(training_data) >= context_window+1:
    for index in range(len(training_data) - context_window):
        feature_list.append(tokenized_training_data[index : context_window+index])
        target_list.append(tokenized_training_data[index+1 : index+context_window+1])
else:
    print("Training data is too small for context window")


feature_tensor = torch.tensor(feature_list)
target_tensor = torch.tensor(target_list)

vocab_size = len(unique_chars)
seq_len = 128

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.token_embedding_matrix = nn.Embedding(vocab_size, n_embed)
        self.n_embed = n_embed
        self.attention_layer = nn.MultiheadAttention(n_embed, num_heads=1, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed)
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)
    

    def get_sinusoidal_encodings(self, seq_len, device):
        positions = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.n_embed, 2, device=device) * (-math.log(10000.0) / self.n_embed))
        pe = torch.zeros(seq_len, self.n_embed, device=device)
        pe[:, 0::2] = torch.sin(positions*div_term)
        pe[:, 1::2] = torch.cos(positions*div_term[:self.n_embed//2])
        return pe.unsqueeze(0)


    def forward(self, feature_tensor):
        token_embedding = self.token_embedding_matrix(feature_tensor)
        _, number_of_token = feature_tensor.shape
        position_embedding = self.get_sinusoidal_encodings(number_of_token, feature_tensor.device)
        feature_tensor = token_embedding + position_embedding
        mask = torch.triu(
            torch.ones(
                (number_of_token, number_of_token),
                device=feature_tensor.device,
                dtype=torch.bool
            ),
            diagonal=1
        )
        attention_output, _ = self.attention_layer(
            feature_tensor,
            feature_tensor,
            feature_tensor,
            attn_mask=mask
        )
        feature_tensor = attention_output + feature_tensor
        feed_forward_result = self.feed_forward(feature_tensor)
        feature_tensor = feed_forward_result + feature_tensor
        logits = self.lm_head(feature_tensor)
        return logits
    


model = TinyTransformer(vocab_size=len(unique_chars), n_embed=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def save_model_weights(model, file_path="model_weights_v2.pth"):
    torch.save(model.state_dict(), file_path)
    print(f"Model weight save to path: {file_path}")


def train_llm():
    for epoch in range(1000):
        logits = model(feature_tensor)
        loss = F.cross_entropy(logits.view(-1, len(char_to_index_map)), target_tensor.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            print("=====================================================")
            print(infer_during_training("Elias loved his craft deeply"))
            print("=====================================================")
        
    save_model_weights(model)


def load_model_weight(model, file_path="model_weights_v2.pth"):
    model.load_state_dict(torch.load(file_path))
    return model


def infer(sequence):
    context = torch.tensor([encode(sequence)])
    model = TinyTransformer(vocab_size=len(unique_chars), n_embed=64)
    model = load_model_weight(model)
    model.eval()
    with torch.no_grad():
        for _ in range(200):
            logits = model(context)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
    result = decode(context[0].tolist())
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
