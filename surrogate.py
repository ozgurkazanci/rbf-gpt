# -*- coding: utf-8 -*-
"""TurboRBF ile simülatör vekili (surrogate) — iki hattın birleştiği yer.

Fikir: Spectre koşusu saniyeler sürer; tasarım uzayını taramak için binlerce
koşu gerekir. Az sayıda gerçek koşuyla (transistör boyutu -> ölçüm) eğitilen
bir RBF ağı, aradaki noktaları mikrosaniyede tahmin eder. Ajan böylece
yüzlerce adayı vekille eleyip yalnızca umut vaat edenleri gerçekten simüle
eder.

RBF ağları bu iş için klasik tercihtir: az örnekle düzgün (smooth) yüzeyleri
iyi kurarlar. Buradaki ağ, depodaki ``TurboRBF``tir — yani mimari
çalışmanız doğrudan çip tasarımına hizmet eder.

Kullanım:

    # 1) Veri topla (gercek Spectre ile; deneme icin --simulator mock)
    python agent.py collect --n 40 --out dataset.json

    # 2) Vekili egit
    python surrogate.py train --data dataset.json --out surrogate.pt

    # 3) Ajani vekille kostur (Spectre'a hic dokunmadan), sonucu dogrula
    python agent.py size-inverter --simulator surrogate --model surrogate.pt \
        --target-vm 0.6 --verify
"""

import argparse
import json
import math

import torch

from fast_rbf import TurboRBF

CIKISLAR = ["vm", "gain"]          # tahmin edilen büyüklükler


# ---------------------------------------------------------------------------
# Veri hazırlığı
# ---------------------------------------------------------------------------

def load_dataset(paths, outputs=CIKISLAR):
    """agent.py'nin ürettiği JSON dosyalarını (X, Y) tensörlerine çevirir."""
    kayitlar = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        kayitlar += d.get("adimlar", d if isinstance(d, list) else [])
    veri = [k for k in kayitlar
            if k.get("vm") is not None and k.get("wn_nm") and k.get("wp_nm")]
    if not veri:
        raise SystemExit("Kullanilabilir olcum yok — once 'agent.py collect'"
                         " calistirin.")
    # Öznitelikler log ölçekte: genişlikler çarpımsal büyüklüklerdir.
    X = torch.tensor([[math.log(k["wn_nm"]), math.log(k["wp_nm"]),
                       math.log(k.get("l_nm") or 60.0)] for k in veri],
                     dtype=torch.float32)
    Y = torch.tensor([[float(k.get(o) or 0.0) for o in outputs]
                      for k in veri], dtype=torch.float32)
    return X, Y, veri


def normalize(X, Y, norm=None):
    """Öznitelik/hedef standardizasyonu; norm verilmezse hesaplanır."""
    if norm is None:
        xs = X.std(0)
        ys = Y.std(0)
        norm = {"xm": X.mean(0), "xs": torch.where(xs > 1e-9, xs,
                                                   torch.ones_like(xs)),
                "ym": Y.mean(0), "ys": torch.where(ys > 1e-9, ys,
                                                   torch.ones_like(ys))}
    return ((X - norm["xm"]) / norm["xs"],
            (Y - norm["ym"]) / norm["ys"], norm)


# ---------------------------------------------------------------------------
# Eğitim
# ---------------------------------------------------------------------------

def train(X, Y, centers=None, rank=8, groups=8, active=2, epochs=3000,
          lr=1e-2, val_frac=0.25, seed=0, weight_decay=1e-4, verbose=True):
    """Vekili eğitir; en iyi doğrulama hatasındaki ağırlıkları döndürür.

    Küçük veri kümelerinde (30-60 nokta) ezberleme riski yüksek olduğundan
    ayrı bir doğrulama kümesi tutulur ve en iyi durum saklanır.
    """
    torch.manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = max(2, int(n * val_frac))
    vi, ti = perm[:n_val], perm[n_val:]

    # Merkez sayısı veriyle ölçeklenir: 30 noktaya 32 merkez ezberlemektir.
    if centers is None:
        centers = max(8, len(ti) // 3)
    groups = min(groups, max(2, centers // 2))
    active = min(active, groups)
    if verbose:
        print(f"  yapi: {centers} merkez, {groups} grup, {active} aktif, "
              f"rank {rank}")

    Xn, Yn, norm = normalize(X, Y)
    Xt, Yt, Xv, Yv = Xn[ti], Yn[ti], Xn[vi], Yn[vi]

    model = TurboRBF(X.shape[1], centers, Y.shape[1], rank=rank,
                     num_groups=groups, active_groups=active)
    model.init_from_data(Xt)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    lossf = torch.nn.MSELoss()

    en_iyi, en_iyi_ep, en_iyi_durum = float("inf"), 0, None
    for ep in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xt), Yt)
        loss.backward()
        opt.step()
        if ep % 25 == 0 or ep == epochs - 1:
            with torch.no_grad():
                vloss = lossf(model(Xv), Yv).item()
            if vloss < en_iyi:
                en_iyi, en_iyi_ep = vloss, ep
                en_iyi_durum = {k: v.detach().clone()
                                for k, v in model.state_dict().items()}
            if verbose and ep % 500 == 0:
                print(f"  epoch {ep:5d} | egitim {loss.item():.5f} | "
                      f"dogrulama {vloss:.5f}")
    if en_iyi_durum:
        model.load_state_dict(en_iyi_durum)
    if verbose:
        print(f"  en iyi dogrulama: {en_iyi:.5f} (epoch {en_iyi_ep})")
    return model, norm, {"val_idx": vi.tolist(), "train_idx": ti.tolist(),
                         "centers": centers, "groups": groups,
                         "active": active}


def evaluate(model, norm, X, Y, idx=None, outputs=CIKISLAR):
    """Gerçek birimlerde (volt vb.) hata ölçümleri döndürür."""
    if idx is not None:
        X, Y = X[idx], Y[idx]
    Xn, _, _ = normalize(X, Y, norm)
    with torch.no_grad():
        pred = model(Xn) * norm["ys"] + norm["ym"]
    hata = (pred - Y).abs()
    sonuc = {}
    for j, ad in enumerate(outputs):
        y = Y[:, j]
        ss_tot = ((y - y.mean()) ** 2).sum()
        ss_res = ((pred[:, j] - y) ** 2).sum()
        # Sabit çıkışta R2 tanımsızdır (varyans yok) — sıfır yazmak yanıltır.
        r2 = (1 - ss_res / ss_tot).item() if ss_tot > 1e-12 else None
        sonuc[ad] = {"mae": hata[:, j].mean().item(),
                     "max": hata[:, j].max().item(), "r2": r2}
    return sonuc


def _r2s(v):
    return "sabit (varyans yok)" if v is None else f"{v:.4f}"


def linear_baseline(X, Y, tr, va):
    """Doğrusal en küçük kareler — RBF'in hakkını vermesi için kıyas."""
    A = torch.cat([X[tr], torch.ones(len(tr), 1)], 1)
    w = torch.linalg.lstsq(A, Y[tr]).solution
    Av = torch.cat([X[va], torch.ones(len(va), 1)], 1)
    return (Av @ w - Y[va]).abs().mean(0)


# ---------------------------------------------------------------------------
# Kayıt / yükleme ve ajan arayüzü
# ---------------------------------------------------------------------------

def save(path, model, norm, cfg, outputs=CIKISLAR):
    torch.save({"state_dict": model.state_dict(), "norm": norm,
                "cfg": cfg, "outputs": outputs}, path)


def load(path):
    d = torch.load(path, weights_only=False)
    c = d["cfg"]
    model = TurboRBF(c["in_dim"], c["centers"], c["out_dim"], rank=c["rank"],
                     num_groups=c["groups"], active_groups=c["active"])
    model.load_state_dict(d["state_dict"])
    model.eval()
    return model, d["norm"], d["outputs"]


class SurrogateTools:
    """CadenceTools ile aynı arayüz — ajan farkı hissetmez, sadece hızlanır."""

    def __init__(self, model_path="surrogate.pt"):
        self.model, self.norm, self.outputs = load(model_path)
        self.path = model_path
        self.calls = 0

    def pdk_info(self):
        return {"kaynak": f"vekil model ({self.path})",
                "ciktilar": ", ".join(self.outputs)}

    def measure_inverter(self, wn_nm, wp_nm, l_nm=60):
        self.calls += 1
        x = torch.tensor([[math.log(wn_nm), math.log(wp_nm), math.log(l_nm)]],
                         dtype=torch.float32)
        xn = (x - self.norm["xm"]) / self.norm["xs"]
        with torch.no_grad():
            y = (self.model(xn) * self.norm["ys"] + self.norm["ym"])[0]
        res = {"wn_nm": wn_nm, "wp_nm": wp_nm, "l_nm": l_nm,
               "vekil": True, "sure_s": 0.0}
        res.update({ad: y[j].item() for j, ad in enumerate(self.outputs)})
        return res

    def predict_many(self, wn_nm, wp_nm, l_nm=60):
        """Binlerce adayı tek matris çarpımında değerlendirir.

        wn_nm/wp_nm: eşit uzunlukta listeler. Dönen sözlükte her çıkış
        için değer listesi bulunur. Toplu tarama (screen) bunun sayesinde
        nokta başına döngü kurmadan saniyeler içinde biter.
        """
        self.calls += len(wn_nm)
        x = torch.tensor(
            [[math.log(a), math.log(b), math.log(l_nm)]
             for a, b in zip(wn_nm, wp_nm)], dtype=torch.float32)
        xn = (x - self.norm["xm"]) / self.norm["xs"]
        with torch.no_grad():
            y = self.model(xn) * self.norm["ys"] + self.norm["ym"]
        return {ad: y[:, j].tolist() for j, ad in enumerate(self.outputs)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="TurboRBF simulator vekili")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("train", help="veriden vekil modeli egit")
    sp.add_argument("--data", nargs="+", required=True)
    sp.add_argument("--out", default="surrogate.pt")
    sp.add_argument("--centers", type=int, default=None,
                    help="varsayilan: veri boyutuna gore otomatik (n/3)")
    sp.add_argument("--rank", type=int, default=8)
    sp.add_argument("--groups", type=int, default=8)
    sp.add_argument("--active", type=int, default=2)
    sp.add_argument("--epochs", type=int, default=3000)
    sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("check", help="egitilmis vekili veri uzerinde sina")
    sp.add_argument("--model", default="surrogate.pt")
    sp.add_argument("--data", nargs="+", required=True)

    args = p.parse_args()

    if args.cmd == "check":
        model, norm, outputs = load(args.model)
        X, Y, _ = load_dataset(args.data, outputs)
        m = evaluate(model, norm, X, Y, outputs=outputs)
        print(f"{X.shape[0]} nokta uzerinde:")
        for ad, s in m.items():
            print(f"  {ad:5s} | MAE {s['mae']:.4f} | max {s['max']:.4f} "
                  f"| R2 {_r2s(s['r2'])}")
        return

    X, Y, veri = load_dataset(args.data)
    print(f"Veri: {X.shape[0]} nokta, {X.shape[1]} oznitelik -> "
          f"{Y.shape[1]} cikis ({', '.join(CIKISLAR)})")
    model, norm, split = train(X, Y, centers=args.centers, rank=args.rank,
                               groups=args.groups, active=args.active,
                               epochs=args.epochs, seed=args.seed)

    tr, va = split["train_idx"], split["val_idx"]
    print(f"\nEgitim ({len(tr)} nokta):")
    for ad, s in evaluate(model, norm, X, Y, tr).items():
        print(f"  {ad:5s} | MAE {s['mae']:.4f} | R2 {_r2s(s['r2'])}")
    print(f"Dogrulama ({len(va)} nokta — model bunlari hic gormedi):")
    va_m = evaluate(model, norm, X, Y, va)
    for ad, s in va_m.items():
        print(f"  {ad:5s} | MAE {s['mae']:.4f} | max {s['max']:.4f} "
              f"| R2 {_r2s(s['r2'])}")

    lin = linear_baseline(X, Y, tr, va)
    print("Dogrusal kiyas (ayni dogrulama noktalari):")
    for j, ad in enumerate(CIKISLAR):
        rbf_mae, lin_mae = va_m[ad]["mae"], lin[j].item()
        if lin_mae < rbf_mae:
            print(f"  {ad:5s} | dogrusal MAE {lin_mae:.4f} < TurboRBF "
                  f"{rbf_mae:.4f} — bu cikis icin yuzey neredeyse dogrusal, "
                  f"RBF'in avantaji yok")
        else:
            kat = lin_mae / rbf_mae if rbf_mae > 1e-9 else float("inf")
            print(f"  {ad:5s} | dogrusal MAE {lin_mae:.4f} -> TurboRBF "
                  f"{kat:.1f}x daha iyi")

    save(args.out, model, norm,
         {"in_dim": X.shape[1], "out_dim": Y.shape[1],
          "centers": split["centers"], "rank": args.rank,
          "groups": split["groups"], "active": split["active"]},
         CIKISLAR)
    print(f"\nVekil kaydedildi: {args.out}")
    print(f"Vm dogrulama hatasi: {va_m['vm']['mae']*1000:.1f} mV")


if __name__ == "__main__":
    main()
