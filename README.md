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
