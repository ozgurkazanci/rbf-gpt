"""RBF-GPT: a minimal GPT whose feed-forward blocks are RBF networks.

Each transformer block keeps standard causal self-attention, but the usual
MLP (Linear -> GELU -> Linear) is replaced by an RBF layer:

* ``ffn="naive"``  uses ``NaiveRBF``  -- every token is compared against every
  center in the full embedding dimension.
* ``ffn="turbo"``  uses ``TurboRBF``  -- the sparsely-gated low-rank RBF, which
  is the fast path this repo is about.

Run the built-in demo (trains a tiny char-level model on an embedded corpus
and compares the two FFN variants):

    python -m fast_rbf.rbf_gpt
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import NaiveRBF, TurboRBF


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = (B, T, self.n_head, C // self.n_head)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class RBFBlock(nn.Module):
    """Transformer block with an RBF network as the feed-forward layer."""

    def __init__(self, n_embd, n_head, num_centers, ffn="turbo",
                 rank=16, num_groups=32, active_groups=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        if ffn == "turbo":
            self.ffn = TurboRBF(n_embd, num_centers, n_embd, rank=rank,
                                num_groups=num_groups,
                                active_groups=active_groups)
        elif ffn == "naive":
            self.ffn = NaiveRBF(n_embd, num_centers, n_embd)
        else:
            raise ValueError(f"unknown ffn kind: {ffn!r}")

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        B, T, C = x.shape
        # RBF layers operate on flat (tokens, features) batches.
        x = x + self.ffn(self.ln2(x).view(B * T, C)).view(B, T, C)
        return x


class RBFGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=128, n_head=4,
                 n_layer=2, num_centers=512, ffn="turbo", **rbf_kwargs):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            RBFBlock(n_embd, n_head, num_centers, ffn=ffn, **rbf_kwargs)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

    def warm_start(self, idx):
        """Warm-start every RBF FFN (naive or turbo) from real activations."""
        with torch.no_grad():
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for block in self.blocks:
                if hasattr(block.ffn, "init_from_data"):
                    flat = block.ln2(x + block.attn(block.ln1(x)))
                    block.ffn.init_from_data(flat.view(-1, flat.shape[-1]))
                x = block(x)


# --------------------------------------------------------------------------
# Demo: train tiny char-level models and compare naive vs turbo RBF FFNs.
# --------------------------------------------------------------------------

_CORPUS = """
In the beginning the network was slow, and the centers were many.
Every token asked every center for its distance, and the centers answered
one by one, and the clock ran long. Then the centers were gathered into
groups, and each group raised a prototype, and the tokens spoke only to
the nearest groups, and the clock ran short. The radial basis endured;
only the counting changed. What decays exponentially may be skipped
without sorrow, for its contribution was already nothing.
""" * 20


def _batches(data, block_size, batch_size, device, gen=None):
    # A dedicated generator keeps the batch stream identical across model
    # variants even though model construction consumes global RNG.
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


def run_demo(steps=200, block_size=64, batch_size=16, device="cpu"):
    torch.manual_seed(0)
    chars = sorted(set(_CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in _CORPUS], dtype=torch.long)
    print(f"corpus: {len(data)} chars, vocab {len(chars)}")

    results = {}
    for kind in ("naive", "turbo"):
        torch.manual_seed(0)
        model = RBFGPT(len(chars), block_size, ffn=kind).to(device)
        gen = torch.Generator().manual_seed(123)
        xb, yb = _batches(data, block_size, batch_size, device, gen)
        # Warm-start both variants so the quality comparison is fair.
        model.warm_start(xb)

        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        t0, loss = time.perf_counter(), None
        for step in range(steps):
            xb, yb = _batches(data, block_size, batch_size, device, gen)
            _, loss = model(xb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        dt = time.perf_counter() - t0
        results[kind] = (dt / steps * 1e3, loss.item())
        print(f"{kind:>5} RBF-GPT: {dt / steps * 1e3:7.1f} ms/step  "
              f"final loss {loss.item():.3f}")

    ms_naive, _ = results["naive"]
    ms_turbo, _ = results["turbo"]
    print(f"training speedup (turbo vs naive FFN): {ms_naive / ms_turbo:.2f}x")


if __name__ == "__main__":
    run_demo()
