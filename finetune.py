# -*- coding: utf-8 -*-
"""Kendi modeline geçiş — uzman gösterimlerinden eğitim verisi üretimi.

Fikir (davranis klonlama / expert iteration): Qwen gibi genel modeller devre
fizigini bilmiyordu (orani sabit tuttu, hedefe ulasamadi). Ama depodaki
kurallı RuleBasedSizer BU ISI DOGRU YAPIYOR — orani kusatip ikiye bolerek
4-5 olcumde hedefe varyor. Bu uzmani "ogretmen" olarak kullanip cok sayida
cozulmus senaryo uretiriz; her adim bir "durum -> dogru arac cagrisi"
ornegidir. Bir taban model (Qwen/Llama) bu gosterimlerle ince ayar edilince
uzmanin akil yurutmesini icsellestirir — artik sizin modeliniz olur.

Uretilen veri, arac-cagrili sohbet (messages) bicimindedir; hem Ollama
ince ayarina hem HuggingFace/LoRA egitimine dogrudan beslenebilir.

Kullanım:
    # 1) Uzman gosterimleri uret (Cadence gerekmez; --simulator mock hizli)
    python finetune.py dataset --n-tasks 200 --out sft.jsonl --simulator mock

    # 2a) HAFIF YOL — Ollama Modelfile ile sistem-promptu gomulu model
    #     (egitim yok; uzman stratejisini prompt olarak sabitler)
    python finetune.py modelfile --out Modelfile
    #     ollama create rbf-designer -f Modelfile

    # 2b) GERCEK YOL — LoRA ince ayar (GPU/torch gerekir; komut basılır)
    python finetune.py lora-cmd --data sft.jsonl
"""

import argparse
import json
import random

from agent import MockTools, RuleBasedSizer
from llm_controller import SISTEM_TALIMATI, make_tools


def _expert_trajectory(tools, target_vm, wn=200.0, tol=0.005):
    """Uzman kontrolcüyü koşturur; her ölçümü bir (durum, doğru-hamle) yapar.

    Dönen: bu görevin sohbet biçimli eğitim mesajları (system + user +
    uzmanın attığı her tool-call ve dönen tool-result).
    """
    sizer = RuleBasedSizer(tools, wn_nm=wn)
    sizer.solve(target_vm, tol=tol)
    if not any(h.get("vm") is not None for h in sizer.history):
        return None

    gorev = f"Eviriciyi Vm={target_vm:g}V olacak sekilde boyutlandir."
    messages = [{"role": "system", "content": SISTEM_TALIMATI},
                {"role": "user", "content": gorev}]
    for h in sizer.history:
        if h.get("vm") is None:
            continue
        # Uzmanın attığı hamle = doğru tool-call (modelin taklit edecegi hedef)
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {
                "name": "measure_inverter",
                "arguments": {"wn_nm": h["wn_nm"], "wp_nm": h["wp_nm"]}}}]})
        messages.append({
            "role": "tool", "name": "measure_inverter",
            "content": json.dumps({k: h[k] for k in ("wn_nm", "wp_nm", "vm")
                                   if k in h})})
    # Kapanış: uzman sonucu — modelin öğreneceği "bitiş" davranışı
    son = min((h for h in sizer.history if h.get("vm") is not None),
              key=lambda h: abs(h["vm"] - target_vm))
    ratio = son["wp_nm"] / son["wn_nm"]
    ulasti = abs(son["vm"] - target_vm) <= tol
    ozet = (f"Wn={son['wn_nm']:g} nm, Wp={son['wp_nm']:g} nm "
            f"(oran {ratio:.3f}) ile Vm={son['vm']:.4f} V olctum; "
            + ("hedefe ulasildi." if ulasti
               else "bu topolojide hedefe ulasilamadi, en yakin nokta bu."))
    messages.append({"role": "assistant", "content": ozet})
    return {"messages": messages, "hedef": target_vm, "olcum": len(messages)}


def build_dataset(n_tasks, simulator="mock", vm_min=0.45, vm_max=0.75,
                  wn_choices=(120, 160, 200, 280, 400), seed=0):
    tools = MockTools() if simulator == "mock" else make_tools(simulator)
    rng = random.Random(seed)
    veri = []
    for i in range(n_tasks):
        target = round(rng.uniform(vm_min, vm_max), 3)
        wn = float(rng.choice(wn_choices))
        traj = _expert_trajectory(tools, target, wn=wn)
        if traj:
            veri.append(traj)
    return veri


MODELFILE_SABLONU = '''# rbf-designer — uzman stratejisi gomulu evirici tasarim modeli
# Kullanim: ollama create rbf-designer -f Modelfile
#           python llm_controller.py --task "..." --llm rbf-designer
FROM {base}

PARAMETER temperature 0.3

SYSTEM """{system}"""
'''


def lora_command(data):
    return f"""# LoRA ince ayar — GPU'lu bir makinede (Colab T4 / kiralik 4090):
pip install "unsloth[cu121]" trl datasets

python - <<'PY'
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct", max_seq_length=4096, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"])

ds = load_dataset("json", data_files="{data}", split="train")
ds = ds.map(lambda e: {{"text": tok.apply_chat_template(
    e["messages"], tokenize=False)}})

SFTTrainer(model=model, tokenizer=tok, train_dataset=ds,
    args=SFTConfig(per_device_train_batch_size=2, num_train_epochs=3,
        learning_rate=2e-4, output_dir="rbf-designer-lora")).train()

model.save_pretrained_gguf("rbf-designer", tok)   # Ollama'ya alinabilir
PY
# Sonra: ollama create rbf-designer -f rbf-designer/Modelfile"""


def main():
    p = argparse.ArgumentParser(description="Kendi tasarim modeline gecis")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("dataset", help="uzman gosterimlerinden SFT verisi")
    sp.add_argument("--n-tasks", type=int, default=200)
    sp.add_argument("--simulator", choices=["mock", "spectre", "surrogate"],
                    default="mock")
    sp.add_argument("--out", default="sft.jsonl")
    sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("modelfile", help="Ollama Modelfile (egitimsiz yol)")
    sp.add_argument("--base", default="qwen2.5:7b")
    sp.add_argument("--out", default="Modelfile")

    sp = sub.add_parser("lora-cmd", help="LoRA ince ayar komutunu bas")
    sp.add_argument("--data", default="sft.jsonl")

    args = p.parse_args()

    if args.cmd == "dataset":
        veri = build_dataset(args.n_tasks, simulator=args.simulator,
                             seed=args.seed)
        with open(args.out, "w", encoding="utf-8") as f:
            for ornek in veri:
                f.write(json.dumps(ornek, ensure_ascii=False) + "\n")
        toplam_adim = sum(o["olcum"] for o in veri)
        print(f"{len(veri)} gorev, ~{toplam_adim} mesaj -> {args.out}")
        print(f"Ornek gorev: {veri[0]['messages'][1]['content']}")
        n_call = sum(1 for m in veri[0]["messages"]
                     if m.get("tool_calls"))
        print(f"Ilk gorevde uzman {n_call} olcumle cozdu.")
        print(f"\nSonraki: python finetune.py lora-cmd --data {args.out}")
        return

    if args.cmd == "modelfile":
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(MODELFILE_SABLONU.format(base=args.base,
                                             system=SISTEM_TALIMATI))
        print(f"{args.out} yazildi.")
        print(f"  ollama create rbf-designer -f {args.out}")
        print("  python llm_controller.py --task \"...\" --llm rbf-designer")
        return

    if args.cmd == "lora-cmd":
        print(lora_command(args.data))


if __name__ == "__main__":
    main()
