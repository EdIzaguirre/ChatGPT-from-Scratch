import torch
import torch.nn as nn
from torch.nn import functional as F
import platform
import sys

# hyperparameters
block_size = 256
batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
# -------------
# Checking for GPU
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print("NVIDIA/CUDA GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")
# --------------

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


class Head(nn.Module):
    # Creates a single head of self-attention
    def __init__(self, head_size):
        super().__init__()
        # Creating the dense layers for key, query, and value.
        # Necessary to transform inputs to new space, to be trained
        # by model to make useful representations
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Defining a forward pass. Input data x is transformed
        # Masking is done to prevent "looking ahead"
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    # Getting multiple heads of attention in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearirty"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            # Doing projection to the residual highway
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ A transformer block: communication followed by computation"""

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head, the number of heads
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # Here we see skip connections being implemented. Can think of this
        # as forking off, doing some computation, then coming back.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    # Creation of a language model
    def __init__(self):
        super().__init__()
        # Initialize an embedding layer with random values of
        # dimension [vocab_size, vocab_size]
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # Create a similar embedding to encode position as well
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embed)
        # Linear layer, an intermediate layer. We don't want to go straight
        # to the components of the embedding for the logits, we need something
        # more sophisticated.
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx is of dimension [Batch_size, block_size]
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        # Creation of positional embedding, given length of window T
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T,C)

        # Combining token and positional embeddings
        x = tok_emb + pos_emb  # (B,T,C)
        # Run code through blocks of MHA and Dense layers
        x = self.blocks(x)
        # Layer norm
        x = self.ln_f(x)
        # Throw them into that intermediate, linear layer
        logits = self.lm_head(x)

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
            # Crop idx to the last block_size tokens. Because we are doing
            # positional embeddings now, and that embedding table only has
            # block_size rows (possible character tokens), passing in a token
            # higher than the block_size rows would throw an error
            # Wasn't necessary for character embeddings, because we could only
            # generate 26 characters.
            idx_cond = idx[:, -block_size:]

            # Produces a bunch of logits (embeddings), ignore the loss
            # There is no target when doing generation
            logits, loss = self(idx_cond)

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


model = GPTLanguageModel()
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

torch.save(m.state_dict(), 'model_weights.pth')
