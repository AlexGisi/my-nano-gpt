with open("tiny-shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda a: "".join([itoc[x] for x in a])


import torch
import time

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# The important thing to realize is when we train a transformer, we work with chunks
# of the training set -- randomly sampling these chunks with a max length. Call it
# block size.
block_size = 8

# We will train the transformer to predict the next character at each position in the sequence.
print(train_data[: block_size + 1])

# There are eight examples in a chunk of nine characters.
x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"When input is {context} the target is {target}")

torch.manual_seed(1337)
batch_size = 4
block_size = 8


def get_batch(split, device="cpu"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

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
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
# xb, yb = get_batch("train", device=device)
m = BigramLanguageModel(vocab_size).to(device)
out, loss = m(xb, yb)

print(out.shape)
print(loss)

print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)  # usually a good one is 1e-4, but smaller networks can have 1e-3 or bigger
batch_size = 32
start = time.monotonic()
for steps in range(50000):
    xb, yb = get_batch('train', device=device)

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(loss.item())

    if loss.item() < 2.5:
        break
end = time.monotonic()

print(f"Final loss {loss.item()} in {end-start} seconds")

print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))
