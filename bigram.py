import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
block_size = 8
batch_size = 32
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'mps' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Taking stock of all unique character values in the literature
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Creating two dictionaries, one for string to integer (stoi)
# and one for integer to string (itos)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(line):
    return ''.join([itos[i] for i in line])


# Encoding all of our text and placing the resulting
# Python list into a PyTorch tensor. Part of the
# Tokenization aspect of NLP
data = torch.tensor(encode(text), dtype=torch.long)

# Generating train/val split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # Function to create batches of blocks
    data = train_data if split == 'train' else val_data
    # Picks batch_size random numbers from 0 to len(data)-block_size
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Goes one random number at a time, creating a list of data
    # starting from that number up to random_num+block_size
    # Then they are all concatenated horizontally to create a batch
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
# Makes sure that everything in this function is not involved in back-prop
def estimate_loss():
    out = {}
    # Sets model to eval phase
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # Sets model to train phase
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    # Creation of a bigram language model
    def __init__(self, vocab_size):
        super().__init__()
        # Initialize an embedding layer with random values of
        # dimension [vocab_size, vocab_size]
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx is of dimension [Batch_size, block_size]
        logits = self.token_embedding_table(idx)
        # logits (a misnomer) is of dimension
        # [Batch_size, block_size, vocab_size]. This is because
        # each single number is embedded to a vector of length
        # vocab_size, as dictated when creating the embedding layer

        if targets is None:
            loss = 0
        else:
            B, T, C = logits.shape
            # .view() in PyTorch is like .reshape() in TensorFlow
            # Reshaping here to work properly with the cross_entropy
            # function from PyTorch
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Produces a bunch of logits (embeddings), ignore the loss
            # There is no target when doing generation
            logits, loss = self(idx)

            # Get last token from the columns (2nd dimension)
            # This symbolizes the embedding vector for the very
            # last character
            logits = logits[:, -1, :]

            # Do a softmax over all vocab_size components of the
            # embedding vector
            probs = F.softmax(logits, dim=-1)

            # Sample once from the normalized components of the embedding
            # vector. Since it is of size vocab_size, this will correspond
            # to one of the numbers in your vocabulary
            idx_next = torch.multinomial(probs, num_samples=1)

            # Add the new number to your idx, and keep going
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create PyTorch optmizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# Creating a training loop in PyTorch
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
