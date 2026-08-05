# -*- coding: utf-8 -*-
"""
mini_gpt_rbf.py — RBF-Kernel Seçenekli Mini GPT (TurboRBF entegrasyonlu)
========================================================================
Karakter düzeyinde eğitilen, tek dosyalık bir mini transformer dil modeli.

NOVELTY DENEYİNİZ: Attention mekanizması iki modda çalışır:
  --attention softmax : Klasik GPT attention'ı  ->  softmax(Q·Kᵀ / √d)
  --attention rbf     : RBF (Gauss) kernel attention -> softmax(-‖q-k‖² / 2σ²)
                        (σ her attention başı için ÖĞRENİLEBİLİR parametredir,
                         tıpkı klasik RBFN'deki genişlik parametresi gibi!)

Matematiksel bağlantı (sizin RBF geçmişinizle köprü):
  ‖q-k‖² = ‖q‖² + ‖k‖² - 2·q·k
  Yani RBF attention, nokta çarpım attention'ına "norm düzeltmesi" eklenmiş
  halidir. q·k benzerliği yerine Öklid uzaklığına dayalı Gauss benzerliği
  kullanılır — klasik RBFN'de girdinin merkezlere yakınlığını ölçmenizle
  aynı fikir. Burada "merkezler" sabit değil, her token'ın key vektörüdür.

HIZLANDIRMA 1 — "rbf-fast" (varsayılan, SONUÇ BİREBİR AYNI):
  Softmax satır bazında normalize ettiği için -‖q‖²/2σ² terimi sadeleşir:
      softmax_j(-‖q-k_j‖²/2σ²) = softmax_j( q·k_j/σ² - ‖k_j‖²/2σ² )
  Key'lere tek bir ek boyut ekleyip norm düzeltmesini içine gömersek
      q̂ = [q/σ, 1],   k̂ = [k/σ, -‖k‖²/2σ²]   =>   q̂·k̂ = RBF skoru
  RBF attention, PyTorch'un füzyonlu scaled_dot_product_attention
  (Flash/memory-efficient) çekirdeğiyle AYNEN hesaplanabilir. Yaklaşıklama
  yok; sadece T×T skor matrisinin elle kurulup maskelenmesi ortadan kalkar.
  Eski yol --rbf-impl naive ile hâlâ seçilebilir (kıyas/öğretim için).

HIZLANDIRMA 2 — TurboRBF MLP (dosyadaki "2. deney" notunuz):
  --mlp turbo-rbf, her bloktaki MLP'yi fast_rbf.TurboRBF ile değiştirir:
  düşük-ranklı metrik + MoE tarzı kaba-ince merkez yönlendirme + matmul-form
  mesafeler. Model böylece attention'da VE ileri beslemede RBF kullanır.

KULLANIM:
  python mini_gpt_rbf.py --attention softmax             # baseline eğitimi
  python mini_gpt_rbf.py --attention rbf                 # RBF (hızlı yol)
  python mini_gpt_rbf.py --attention rbf --rbf-impl naive  # eski RBF yolu
  python mini_gpt_rbf.py --attention rbf --mlp turbo-rbf   # tam RBF modeli
  python mini_gpt_rbf.py --attention rbf --iters 5000 --device cuda

Veri: klasördeki input.txt kullanılır (kendi Türkçe metninizi koyabilirsiniz).
      Yoksa Tiny Shakespeare otomatik indirilir.

Deney önerisi: Aynı --seed ile iki modu da eğitin, val loss eğrilerini ve
final perplexity'yi karşılaştırın. Sonuçları GitHub'a push'layın (bulut!).
"""

import argparse
import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fast_rbf import TurboRBF
except ImportError:  # dosya repo dışına tek başına kopyalanırsa
    TurboRBF = None

# ----------------------------------------------------------------------------
# 1) AYARLAR
# ----------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="RBF-kernel seçenekli mini GPT")
    p.add_argument("--attention", choices=["softmax", "rbf"], default="softmax",
                   help="Attention tipi: klasik 'softmax' veya 'rbf' kernel")
    p.add_argument("--rbf-impl", choices=["fast", "naive"], default="fast",
                   help="RBF attention yolu: 'fast' (SDPA, birebir aynı sonuç) "
                        "veya 'naive' (elle T×T skor matrisi)")
    p.add_argument("--mlp", choices=["gelu", "turbo-rbf"], default="gelu",
                   help="Blok MLP'si: klasik 'gelu' veya 'turbo-rbf' "
                        "(fast_rbf.TurboRBF ileri besleme katmanı)")
    p.add_argument("--rbf-centers", type=int, default=256,
                   help="turbo-rbf MLP'deki merkez sayısı")
    p.add_argument("--iters", type=int, default=3000, help="Eğitim adımı sayısı")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128,
                   help="Bağlam uzunluğu (kaç karakter geriye bakılır)")
    p.add_argument("--n-embd", type=int, default=192, help="Gömme boyutu")
    p.add_argument("--n-head", type=int, default=6, help="Attention başı sayısı")
    p.add_argument("--n-layer", type=int, default=4, help="Transformer blok sayısı")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", help="'cuda', 'cpu' veya 'auto'")
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--sample-chars", type=int, default=400,
                   help="Eğitim sonunda üretilecek örnek metin uzunluğu")
    return p.parse_args()

# ----------------------------------------------------------------------------
# 2) VERİ: input.txt varsa onu kullan, yoksa Tiny Shakespeare indir
# ----------------------------------------------------------------------------

def load_text():
    if os.path.exists("input.txt"):
        print("Veri: input.txt bulundu, o kullanılıyor.")
        with open("input.txt", "r", encoding="utf-8") as f:
            return f.read()
    url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
           "data/tinyshakespeare/input.txt")
    print("Veri: input.txt yok, Tiny Shakespeare indiriliyor...")
    text = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return text

# ----------------------------------------------------------------------------
# 3) TOKENIZER (karakter düzeyi — en basit BPE öncesi başlangıç)
# ----------------------------------------------------------------------------

class CharTokenizer:
    """Her benzersiz karaktere bir tam sayı atar. Basit ama öğretici."""

    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

# ----------------------------------------------------------------------------
# 4) ATTENTION — projenin kalbi ve sizin novelty noktanız
# ----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Çok başlı, nedensel (causal) self-attention.

    mode="softmax":  skor = (q · kᵀ) / √d          -> klasik GPT
    mode="rbf":      skor = -‖q - k‖² / (2σ²)      -> Gauss kerneli
                     σ her baş için öğrenilebilir (log_sigma olarak tutulur
                     ki σ her zaman pozitif kalsın).

    Her iki modda da skorlara causal mask uygulanır ve satır bazında softmax
    alınır; yani RBF modunda elde edilen şey NORMALİZE Gauss kernel
    ağırlıklarıdır — klasik RBFN çıkış katmanının attention'a uyarlanmış hali.

    RBF modunda iki eşdeğer yol vardır:
      impl="naive": T×T uzaklık matrisi elle kurulur (öğretici, yavaş).
      impl="fast":  norm düzeltmesi key'lere ek bir boyut olarak gömülür ve
                    füzyonlu scaled_dot_product_attention çağrılır. Softmax
                    -‖q‖²/2σ² terimini sadeleştirdiği için sonuç BİREBİR
                    aynıdır; sadece hesap daha hızlıdır.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.mode = cfg.attention
        self.impl = getattr(cfg, "rbf_impl", "fast")
        self.dropout_p = cfg.dropout

        # Q, K, V projeksiyonları tek matriste (verimlilik için)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        if self.mode == "rbf":
            # Baş başına öğrenilebilir genişlik: σ = exp(log_sigma)
            # Başlangıç: σ ≈ √(head_dim)'in karekökü ölçeği — skorların
            # softmax için makul aralıkta başlamasını sağlar.
            init = 0.5 * math.log(self.head_dim)
            self.log_sigma = nn.Parameter(torch.full((cfg.n_head,), init))

        # Causal mask: gelecekteki token'lara bakmayı yasaklar (naive yol için;
        # fast yol maskelemeyi SDPA'nın is_causal'ına bırakır)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, zaman(token), kanal
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.mode == "rbf" and self.impl == "fast":
            # ‖q-k‖²'nin -‖q‖²/2σ² kısmı softmax'ta sadeleşir; kalan
            # q·k/σ² - ‖k‖²/2σ² skoru, bir boyut genişletilmiş q̂·k̂'ya eşittir.
            sigma = torch.exp(self.log_sigma).view(1, -1, 1, 1)
            qs, ks = q / sigma, k / sigma
            k2 = (ks * ks).sum(-1, keepdim=True)            # (B, H, T, 1)
            q_aug = torch.cat([qs, torch.ones_like(qs[..., :1])], dim=-1)
            k_aug = torch.cat([ks, -0.5 * k2], dim=-1)
            y = F.scaled_dot_product_attention(
                q_aug, k_aug, v, is_causal=True, scale=1.0,
                dropout_p=self.dropout_p if self.training else 0.0)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_drop(self.proj(y))

        if self.mode == "softmax":
            # Klasik: ölçekli nokta çarpım
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        else:
            # RBF (naive): negatif kare Öklid uzaklığı / (2σ²)
            # cdist yerine açılımı kullanıyoruz (autograd dostu ve hızlı):
            # ‖q-k‖² = ‖q‖² + ‖k‖² - 2 q·k
            q2 = (q * q).sum(-1, keepdim=True)             # (B, H, T, 1)
            k2 = (k * k).sum(-1, keepdim=True).transpose(-2, -1)  # (B, H, 1, T)
            d2 = q2 + k2 - 2.0 * (q @ k.transpose(-2, -1)) # (B, H, T, T)
            d2 = d2.clamp(min=0.0)  # sayısal güvenlik (negatif sıfırlar)
            sigma2 = torch.exp(2.0 * self.log_sigma).view(1, -1, 1, 1)
            att = -d2 / (2.0 * sigma2)

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                        # (B, H, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # başları birleştir
        return self.resid_drop(self.proj(y))

# ----------------------------------------------------------------------------
# 5) TRANSFORMER BLOĞU ve MODEL
# ----------------------------------------------------------------------------

class MLP(nn.Module):
    """Her bloktaki ileri beslemeli ağ (klasik GELU'lu sürüm)."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TurboRBFMLP(nn.Module):
    """MLP yerine geçen TurboRBF ileri besleme katmanı ("2. deney").

    Token'lar (B·T, C) olarak düzleştirilip seyrek-geçitli düşük-ranklı RBF
    ağından geçirilir; çıkış yine (B, T, C) olur.
    """

    def __init__(self, cfg):
        super().__init__()
        if TurboRBF is None:
            raise ImportError(
                "--mlp turbo-rbf için fast_rbf paketi gerekli; bu dosyayı "
                "rbf-gpt deposunun kökünden çalıştırın.")
        self.rbf = TurboRBF(cfg.n_embd, cfg.rbf_centers, cfg.n_embd,
                            rank=16, num_groups=16, active_groups=4)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        return self.drop(self.rbf(x.view(B * T, C)).view(B, T, C))


class Block(nn.Module):
    """Pre-LayerNorm transformer bloğu: x + Attn(LN(x)), sonra x + MLP(LN(x))"""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        if getattr(cfg, "mlp", "gelu") == "turbo-rbf":
            self.mlp = TurboRBFMLP(cfg)
        else:
            self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)   # kelime gömme
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # konum gömme
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Verilen bağlamdan devam ederek metin üretir."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        self.train()
        return idx

# ----------------------------------------------------------------------------
# 6) EĞİTİM
# ----------------------------------------------------------------------------

def get_batch(data, cfg, device):
    """Rastgele (girdi, hedef) çiftleri örnekler. Hedef = girdinin 1 kaydırılmışı."""
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([data[i:i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + cfg.block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device, iters=50):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = get_batch(data, cfg, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    cfg = get_args()
    torch.manual_seed(cfg.seed)

    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device
    extra = f" ({cfg.rbf_impl})" if cfg.attention == "rbf" else ""
    print(f"Cihaz: {device} | Attention: {cfg.attention.upper()}{extra} | "
          f"MLP: {cfg.mlp}")

    # --- Veri hazırlığı ---
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"Korpus: {len(text):,} karakter | Sözlük: {tok.vocab_size} karakter")

    # --- Model ---
    model = MiniGPT(cfg, tok.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametre sayısı: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # --- Döngü ---
    log_path = f"log_{cfg.attention}.csv"
    with open(log_path, "w") as f:
        f.write("iter,train_loss,val_loss\n")

    t0 = time.time()
    for it in range(cfg.iters + 1):
        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            ppl = math.exp(losses["val"])
            elapsed = time.time() - t0
            msg = (f"adım {it:5d} | train {losses['train']:.4f} | "
                   f"val {losses['val']:.4f} | perplexity {ppl:6.2f} | {elapsed:6.1f}s")
            if cfg.attention == "rbf":
                sigmas = torch.exp(model.blocks[0].attn.log_sigma).tolist()
                msg += " | σ(blok1): " + ",".join(f"{s:.2f}" for s in sigmas)
            print(msg)
            with open(log_path, "a") as f:
                f.write(f"{it},{losses['train']:.4f},{losses['val']:.4f}\n")

        x, y = get_batch(train_data, cfg, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # --- Checkpoint kaydet (bunu GitHub/HF Hub'a push'layın -> bulut!) ---
    ckpt_path = f"ckpt_{cfg.attention}.pt"
    torch.save({"model": model.state_dict(), "config": vars(cfg),
                "stoi": tok.stoi}, ckpt_path)
    print(f"\nModel kaydedildi: {ckpt_path} | Log: {log_path}")

    # --- Örnek üretim ---
    print("\n--- ÜRETİLEN ÖRNEK METİN ---")
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(start, max_new_tokens=cfg.sample_chars)
    print(tok.decode(sample[0].tolist()))


if __name__ == "__main__":
    main()
