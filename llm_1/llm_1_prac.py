import torch
import torch.nn as nn
import torch.nn.functional as F


# training_data = """
# Once upon a time, in a small village by the river, there lived a kind baker. 
# Every morning he woke up early to bake bread, cakes, and sweet rolls for the people. 
# The children loved the smell of fresh bread, and they often gathered by the shop to watch him work. 
# One day, a stray cat wandered into the bakery. The baker gave it some warm milk and a cozy place to sleep. 
# From then on, the cat stayed by his side, greeting the villagers every morning with a cheerful meow. 
# """

training_data = """
    In a quiet valley surrounded by soft, rolling hills, there was a bustling town known for its colorful markets and friendly people. 
    Every sunrise, merchants filled the streets with fruits, spices, fabrics, and trinkets from distant lands. 
    Children hurried across cobblestone paths on their way to lessons, waving to neighbors and laughing in the crisp morning air. 
    At the edge of the town square stood a tall clock tower, its bells ringing proudly to welcome each new day. 

    In this town lived a humble shoemaker named Elias. He worked in a tiny workshop just beside the clock tower, 
    crafting sturdy boots and elegant slippers for anyone who came through his door. 
    Elias loved his craft deeply, and though he did not have much, he always offered a warm smile and a kind word. 
    He enjoyed listening to stories from travelers, especially those who spoke of oceans, deserts, and kingdoms beyond the horizon. 
    """


unique_chars = set(training_data)
sorted_unique_chars = sorted(unique_chars)

char_to_index_map = {}
for index, char in enumerate(sorted_unique_chars):
    char_to_index_map[char] = index

index_to_char_map = {}
for char, index in char_to_index_map.items():
    index_to_char_map[index] = char

def encode(query: str):
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

if len(training_data) >= context_window + 1:
    for index in range(len(training_data) - context_window):
        feature_list.append(tokenized_training_data[index : index+context_window])
        target_list.append(tokenized_training_data[index+1 : index+context_window+1])
else:
    print("Training data is too small for context window")


feature_tensor = torch.tensor(feature_list)
target_tensor = torch.tensor(target_list)

vocab_size = len(sorted_unique_chars)
max_sequence_length = 256

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.token_embedding_matrix = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_matrix = nn.Embedding(max_sequence_length, n_embed)
        self.attention_layer = nn.MultiheadAttention(n_embed, num_heads=1, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed)
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

        
    def forward(self, feature_tensor):
        token_embedding = self.token_embedding_matrix(feature_tensor)
        _, number_of_token = feature_tensor.shape
        positions = torch.arange(number_of_token, device=feature_tensor.device).unsqueeze(0)
        postion_embedding = self.positional_embedding_matrix(positions)
        feature_tensor = token_embedding + postion_embedding
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
    

model = TinyTransformer(vocab_size=len(char_to_index_map), n_embed=40)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def save_model_weights(model, file_path="tiny_transformer_weights.pth"):
    torch.save(model.state_dict(), file_path)
    print(f"Model weights saved to {file_path}")

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


    
def load_model_weights(model, file_path="tiny_transformer_weights.pth"):
    """Load the model's state dictionary from a file."""
    model.load_state_dict(torch.load(file_path))
    print(f"Model weights loaded from {file_path}")
    return model


def infer(sequence : str):
    context = torch.tensor([encode(sequence)])
    model = TinyTransformer(vocab_size=len(char_to_index_map), n_embed=40)
    model = load_model_weights(model)
    model.eval()    
    with torch.no_grad():
        for _ in range(100):
            logits = model(context)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # next_token = torch.argmax(probs, dim=-1, keepdim=True)
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



