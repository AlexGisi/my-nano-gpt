import time
import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 64
block_size = 256
eval_iters = 200
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
layer_n = 6
dropout = 0.5
embed_dim_n = 384
head_n = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------

with open("tiny-shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda a: "".join([itoc[x] for x in a])

data = torch.tensor(encode(text), dtype=torch.long)

# Create train/test splits
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim_n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim_n, 4 * embed_dim_n),
            nn.ReLU(),
            nn.Linear(4 * embed_dim_n, embed_dim_n),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_dim_n, head_size, bias=False)  # Linear projections.
        self.query = nn.Linear(embed_dim_n, head_size, bias=False)
        self.value = nn.Linear(embed_dim_n, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # Decoder
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Creating multiple independent channels of information helps to aggregate
    different types of data. Because the tokens have lots to talk about!
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim_n, embed_dim_n)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim_n)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim_n)
        self.blocks = nn.Sequential(
            *[Block(embed_dim_n, n_head=head_n) for _ in range(layer_n)]
        )
        self.ln_f = nn.LayerNorm(embed_dim_n)
        self.lm_head = nn.Linear(embed_dim_n, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModel().to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
start = time.monotonic()
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(m)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
end = time.monotonic()

torch.save(m, 'model')
print(f"Trained in {end-start}")

context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
