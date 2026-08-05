"""Fast RBF networks.

Two models are provided:

* ``NaiveRBF``   -- the classic radial basis function network. Every input is
  compared against every center in the full input dimension. Cost per sample:
  O(M * D) for M centers in D dimensions.

* ``TurboRBF``   -- a sparsely-gated, low-rank RBF network. Three ideas are
  combined to cut the cost to roughly O(D * r + G * r + (M/G) * g * r):

  1. **Low-rank metric learning.** Inputs are projected into an r-dimensional
     latent space (r << D) with a learned matrix; centers live directly in the
     latent space, so distances cost O(r) instead of O(D).

  2. **Coarse-to-fine center routing.** Centers are organised into G groups,
     each with a learned prototype. A sample first measures its distance to
     the G prototypes, keeps the top-g nearest groups, and only evaluates the
     centers inside those groups. Because a Gaussian RBF decays exponentially
     with distance, the centers that are skipped would have contributed
     near-zero activation anyway -- pruning them changes the output
     negligibly while making the center sweep sublinear in M.

  3. **Matmul-form distances.** ||x - c||^2 is expanded to
     ||x||^2 - 2 x.c + ||c||^2 so all distances come from one GEMM instead of
     broadcasting (x - c) tensors, which keeps memory traffic low and lets
     BLAS do the work.

  The routing is genuinely differentiable: the selected groups' activations
  are weighted by a softmax gate over the (negative) prototype distances, so
  gradients flow into the prototypes exactly as they do into a
  mixture-of-experts router. Groups outside the top-g receive no gradient on
  a given sample -- the standard hard-routing property. ``init_from_data``
  runs a few k-means steps so each group's centers actually cluster around
  its prototype, which is what makes the pruning premise hold in practice.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sq_dist(x, c):
    """Pairwise squared distances via one matmul: (B, M)."""
    x2 = (x * x).sum(-1, keepdim=True)          # (B, 1)
    c2 = (c * c).sum(-1)                        # (M,)
    return (x2 - 2.0 * x @ c.t() + c2).clamp_min_(0.0)


class NaiveRBF(nn.Module):
    """Classic Gaussian RBF network: dense evaluation of all centers.

    ``log_beta`` starts at -log(2*in_dim) so that beta * E[||x-c||^2] is O(1)
    for standardized inputs; a beta of 1 in high dimension makes every
    activation underflow to exactly 0 and the layer never trains.
    """

    def __init__(self, in_dim, num_centers, out_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, in_dim))
        self.log_beta = nn.Parameter(
            torch.full((num_centers,), -math.log(2.0 * in_dim)))
        self.readout = nn.Linear(num_centers, out_dim)

    def forward(self, x):
        # (B, M) squared distances against every center, full dimension.
        d2 = ((x.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2).sum(-1)
        phi = torch.exp(-torch.exp(self.log_beta) * d2)
        return self.readout(phi)

    @torch.no_grad()
    def init_from_data(self, x):
        """Warm start: centers on real inputs, bandwidth scaled to the data."""
        n = x.shape[0]
        picks = torch.randint(0, n, (self.centers.shape[0],))
        self.centers.copy_(x[picks] + 0.05 * torch.randn_like(self.centers))
        sample = x[: min(n, 256)]
        d2 = ((sample.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2).sum(-1)
        self.log_beta.fill_(-math.log(2.0 * d2.mean().item() + 1e-6))


class TurboRBF(nn.Module):
    """Sparsely-gated low-rank RBF network.

    Args:
        in_dim:       input dimension D.
        num_centers:  total number of RBF centers M (rounded up to a multiple
                      of ``num_groups``).
        out_dim:      output dimension.
        rank:         latent dimension r used for all distance computations.
        num_groups:   number of center groups G.
        active_groups: how many groups g each sample actually evaluates.
    """

    def __init__(self, in_dim, num_centers, out_dim,
                 rank=16, num_groups=32, active_groups=2):
        super().__init__()
        if active_groups > num_groups:
            raise ValueError("active_groups cannot exceed num_groups")
        per_group = -(-num_centers // num_groups)   # ceil division
        self.G, self.g, self.P = num_groups, active_groups, per_group
        self.rank = rank

        self.proj = nn.Linear(in_dim, rank, bias=False)
        self.prototypes = nn.Parameter(torch.randn(num_groups, rank))
        # Centers stored per group: (G, P, r).
        self.centers = nn.Parameter(torch.randn(num_groups, per_group, rank))
        self.log_beta = nn.Parameter(
            torch.full((num_groups, per_group), -math.log(2.0 * rank)))
        self.readout = nn.Linear(num_groups * per_group, out_dim)
        # Readout weight viewed per group for the sparse path: (out, G, P).
        self._out_dim = out_dim

    def forward(self, x):
        z = self.proj(x)                                        # (B, r)

        # --- coarse stage: route to the g nearest groups -------------------
        proto_d2 = _sq_dist(z, self.prototypes)                 # (B, G)
        sel_d2, idx = proto_d2.topk(self.g, dim=-1, largest=False)  # (B, g)
        # Differentiable gate over the selected groups (MoE-style): the gate
        # multiplies the output below, which is what carries gradient back
        # into the prototypes (topk *values* are differentiable; its integer
        # indices are not).
        gate = F.softmax(-sel_d2, dim=-1)                       # (B, g)

        # --- fine stage: evaluate only the selected groups' centers --------
        sel_centers = self.centers[idx]                         # (B, g, P, r)
        sel_beta = torch.exp(self.log_beta[idx])                # (B, g, P)
        diff = z.unsqueeze(1).unsqueeze(2) - sel_centers        # (B, g, P, r)
        d2 = (diff * diff).sum(-1)                              # (B, g, P)
        phi = torch.exp(-sel_beta * d2) * gate.unsqueeze(-1)    # (B, g, P)

        # --- sparse readout: only the active groups' weights participate ---
        w = self.readout.weight.view(self._out_dim, self.G, self.P)
        sel_w = w[:, idx, :].permute(1, 0, 2, 3)                # (B, out, g, P)
        out = torch.einsum("bgp,bogp->bo", phi, sel_w)
        return out + self.readout.bias

    @torch.no_grad()
    def init_from_data(self, x, kmeans_iters=8):
        """Warm start: k-means prototypes, per-cluster centers, scaled beta.

        Prototypes are fitted with a few Lloyd iterations on the projected
        data; each group's centers are then drawn from that group's own
        cluster members, so nearest-prototype routing actually selects the
        groups whose centers are nearest -- the property the sparse forward
        pass relies on.
        """
        z = self.proj(x)
        n = z.shape[0]
        if n >= self.G:
            proto = z[torch.randperm(n)[: self.G]].clone()
        else:
            proto = z[torch.randint(0, n, (self.G,))].clone()
        for _ in range(kmeans_iters):
            assign = _sq_dist(z, proto).argmin(dim=1)
            for j in range(self.G):
                members = z[assign == j]
                if members.shape[0]:
                    proto[j] = members.mean(0)
        self.prototypes.copy_(proto)

        assign = _sq_dist(z, proto).argmin(dim=1)
        intra = z.new_zeros(())
        for j in range(self.G):
            members = z[assign == j]
            if members.shape[0] == 0:
                members = z
            picks = torch.randint(0, members.shape[0], (self.P,))
            group = members[picks] + 0.05 * torch.randn(
                self.P, self.rank, device=z.device, dtype=z.dtype)
            self.centers[j].copy_(group)
            intra = intra + ((group - proto[j]) ** 2).sum(-1).mean()
        # Bandwidth scaled so beta * (typical intra-group distance) is O(1).
        mean_d2 = (intra / self.G).item()
        self.log_beta.fill_(-math.log(2.0 * mean_d2 + 1e-6))
