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

The routing is genuinely differentiable: selected groups' activations are
weighted by a softmax gate over prototype distances (MoE-style), so the
prototypes train with everything else. `init_from_data` warm-starts with a
few k-means steps so each group's centers cluster around its prototype
(nearest-center routing recall ≈ 0.88 vs 0.125 chance), and scales the
bandwidths to the data so activations never underflow.

Measured on CPU (batch 2048, dim 128, 1024 centers; both models
data-warm-started, trained identically):

| model    | forward         | train MSE |
|----------|-----------------|-----------|
| NaiveRBF | ~430–470 ms/iter | 0.056    |
| TurboRBF | ~11–16 ms/iter   | 0.0001   |

≈ **30–40× faster** forward pass, with a better fit in the same number of
steps.

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

### AMD Radeon (DirectML) ile çalıştırma — Windows

Entegre/harici tüm DX12 Radeon'larda (ör. Radeon 780M) çalışır. **Windows
Python'ında** (WSL'de değil, PowerShell'de) kurun:

```powershell
py -3.11 -m venv rbfenv
rbfenv\Scripts\activate
pip install torch-directml
git clone https://github.com/ozgurkazanci/rbf-gpt
cd rbf-gpt
python mini_gpt_rbf.py --attention rbf --device dml --iters 1000
```

`torch-directml`, uyumlu PyTorch sürümünü kendisi kurar (Python 3.11 veya
altı gerekir). Fused attention DML'de desteklenmezse `--rbf-impl naive`
ekleyin — aynı matematik, temel işlemlerle.

## Cadence agent + TurboRBF simulator surrogate

`cadence_bridge/` drives Cadence tools inside WSL from Windows; `agent.py`
closes a design loop on top of it (size an inverter to a target switching
threshold); `surrogate.py` trains TurboRBF on the loop's own measurements so
later searches skip Spectre entirely.

```bash
python -m cadence_bridge.demo check        # tool inventory
python -m cadence_bridge.demo invsim       # inverter VTC on the real PDK
python agent.py size-inverter --target-vm 0.6          # real Spectre loop
python agent.py collect --n 40 --out dataset.json      # sample design space
python surrogate.py train --data dataset.json          # fit TurboRBF
python agent.py size-inverter --simulator surrogate \
    --target-vm 0.6 --verify                           # surrogate + check
```

The surrogate reaches sub-mV accuracy from ~40 samples and answers in ~0.2 ms
versus seconds per Spectre run, so the agent can screen hundreds of candidate
sizings and spend real simulations only on the promising ones.

```bash
# 10.000 adayi ~0.1 saniyede ele, en iyi 5'i gercek Spectre ile dogrula
python agent.py screen --simulator surrogate --target-vm 0.6 --verify
```

### LLM kontrolcüsü — modeli takmak

`llm_controller.py`, kurallı kontrolcünün yerine yerel bir dil modelini
(Ollama) geçirir: görev serbest metindir, model `TOOL_SCHEMAS`'taki araçları
çağırarak ölçer-değerlendirir-yineler. Bugün Ollama'daki açık bir model,
yarın kendi eğittiğiniz model — araç katmanı değişmez.

```powershell
winget install Ollama.Ollama
ollama pull qwen2.5:7b
python llm_controller.py --task "Eviriciyi Vm=0.6V olacak sekilde boyutlandir" --simulator surrogate
```

### Kendi modeliniz — uzman gösteriminden ince ayar

`finetune.py`, kurallı uzman kontrolcüyü öğretmen olarak kullanıp eğitim
verisi üretir (genel modelin bilmediği "oran Vm'i belirler" fiziğini öğretir).
İki yol:

```powershell
# Eğitimsiz: uzman stratejisini sistem-promptu olarak gömen isimli model
python finetune.py modelfile --out Modelfile
ollama create rbf-designer -f Modelfile
python llm_controller.py --task "..." --llm rbf-designer

# Gerçek LoRA: veri üret, Colab'da eğit (finetune_colab.ipynb), GGUF'u Ollama'ya al
python finetune.py dataset --n-tasks 300 --out sft.jsonl
```

### LM Studio ile AMD iGPU'da çıkarım (isteğe bağlı)

Ollama, Radeon 780M gibi iGPU'ları desteklemez (CPU'da koşar). LM Studio ise
Vulkan ile iGPU'yu kullanabilir ve OpenAI-uyumlu yerel sunucu açar:

```powershell
winget install ElementLabs.LMStudio
# LM Studio: modeli yukle (rbf-designer GGUF'u da eklenebilir),
# ayarlardan GPU offload'u acin, Developer sekmesinden Start Server (1234)
python llm_controller.py --backend openai --llm rbf-designer --task "Eviriciyi Vm=0.6V olacak sekilde boyutlandir"
```

Not: üretim hızı (token/s) paylaşımlı RAM bant genişliğiyle sınırlıdır;
iGPU kazancı üretimde mütevazı, uzun prompt işlemede belirgindir.

`finetune_colab.ipynb`'yi Colab'da açın (T4 GPU), `sft.jsonl`'i yükleyin;
notebook Qwen2.5-7B'yi LoRA ile ince ayar edip Ollama'ya alınabilen bir GGUF
üretir. Karşılaştırma: ham qwen2.5:7b oranı 2.0'da sabit tutup Vm≈0.585'te
takılırken, uzman-gösterimli `rbf-designer` oranı 2.0→2.4→2.5 gezerek hedefe
ulaşıyor.

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
