# rbf-gpt

Fast radial basis function (RBF) networks.

## TurboRBF — sparsely-gated low-rank RBF

`fast_rbf/` contains two models:

- **NaiveRBF** — the classic Gaussian RBF network. Every sample is compared
  against every center in the full input dimension: O(M·D) per sample.
- **TurboRBF** — a novel fast variant combining three ideas:
  1. **Low-rank metric learning** — inputs are projected to an r-dimensional
     latent space (r ≪ D); centers live in that space, so each distance costs
     O(r) instead of O(D).
  2. **Coarse-to-fine center routing** — centers are organised into G groups
     with learned prototypes. A sample first routes to its g nearest groups
     (mixture-of-experts style) and only evaluates those groups' centers.
     Skipped centers would have had near-zero Gaussian activation anyway, so
     the output barely changes while the center sweep becomes sublinear in M.
  3. **Matmul-form distances** — ‖x−c‖² is expanded to ‖x‖² − 2x·c + ‖c‖² so
     distances come from a single GEMM instead of broadcast subtraction.

Measured on CPU (batch 2048, dim 128, 1024 centers):

| model    | forward       | train MSE |
|----------|---------------|-----------|
| NaiveRBF | ~426 ms/iter  | 0.0577    |
| TurboRBF | ~11 ms/iter   | 0.0030    |

≈ **40× faster** forward pass, with better fit in the same number of steps
(the learned metric + data-driven warm start help optimisation).

## mini_gpt_rbf.py — RBF-kernel attention'lı mini GPT

`mini_gpt_rbf.py` is a single-file char-level GPT whose attention can run as a
Gaussian (RBF) kernel with a learnable per-head width σ. TurboRBF integration
adds two speedups:

- `--rbf-impl fast` (default) — because softmax normalises per row, the
  −‖q‖²/2σ² term cancels; embedding the key-norm correction as one extra
  dimension (`q̂=[q/σ, 1]`, `k̂=[k/σ, −‖k‖²/2σ²]`) makes RBF attention run on
  the fused `scaled_dot_product_attention` kernel with **bit-for-bit
  equivalent math** (verified: max logit diff 8e-7, max grad diff 1.4e-8).
  ~2.1× faster forward at T=512 on CPU; much larger gains on CUDA where the
  fused kernel avoids materialising the T×T score matrix.
- `--mlp turbo-rbf` — replaces each block's MLP with a `TurboRBF` layer
  (the file's own "experiment #2").

```bash
python mini_gpt_rbf.py --attention rbf                  # fast RBF attention
python mini_gpt_rbf.py --attention rbf --rbf-impl naive # original slow path
python mini_gpt_rbf.py --attention rbf --mlp turbo-rbf  # fully-RBF model
```

## Usage

```bash
pip install -r requirements.txt
python -m fast_rbf.benchmark
```

```python
import torch
from fast_rbf import TurboRBF

model = TurboRBF(in_dim=128, num_centers=1024, out_dim=1,
                 rank=16, num_groups=32, active_groups=4)
model.init_from_data(x_train)   # optional warm start
y = model(x_train)
```
