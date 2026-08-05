"""Benchmark NaiveRBF vs TurboRBF: speed and fit quality.

Run with:  python -m fast_rbf.benchmark
"""

import time

import torch

from .model import NaiveRBF, TurboRBF


def time_forward(model, x, warmup=3, iters=20):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        return (time.perf_counter() - t0) / iters


def fit(model, x, y, steps=100, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return loss.item()


def main():
    torch.manual_seed(0)
    B, D, M, OUT = 2048, 128, 1024, 1

    # Toy regression target: a bumpy nonlinear function of the inputs.
    x = torch.randn(B, D)
    w = torch.randn(D, 4)
    y = torch.sin(x @ w).prod(-1, keepdim=True) + 0.05 * torch.randn(B, 1)

    naive = NaiveRBF(D, M, OUT)
    turbo = TurboRBF(D, M, OUT, rank=16, num_groups=32, active_groups=4)
    turbo.init_from_data(x)

    t_naive = time_forward(naive, x)
    t_turbo = time_forward(turbo, x)

    print(f"batch={B}  dim={D}  centers={M}")
    print(f"NaiveRBF forward: {t_naive * 1e3:8.2f} ms/iter")
    print(f"TurboRBF forward: {t_turbo * 1e3:8.2f} ms/iter")
    print(f"speedup:          {t_naive / t_turbo:8.2f}x")

    # Fit on a subset so the dense baseline finishes in reasonable time on CPU.
    xf, yf = x[:512], y[:512]
    l_naive = fit(naive, xf, yf)
    l_turbo = fit(turbo, xf, yf)
    print(f"final train MSE — naive: {l_naive:.4f}  turbo: {l_turbo:.4f}")


if __name__ == "__main__":
    main()
