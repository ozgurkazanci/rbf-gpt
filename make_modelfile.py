# -*- coding: utf-8 -*-
"""Eğitilmiş GGUF için araç-çağırmalı doğru Modelfile üret.

Ollama, çıplak bir GGUF'tan model oluştururken sohbet şablonunu her zaman
çıkaramaz ve ilkel ``TEMPLATE {{ .Prompt }}`` kalır — bu şablonda tools
desteği olmadığından llm_controller.py araç çağıramaz. Bu betik, taban
modelin (qwen2.5:7b) TEMPLATE/PARAMETER bloklarını
``ollama show --modelfile`` çıktısından alıp yerel GGUF'a bağlar.

Kullanım (zip'i açtığınız, .gguf'un olduğu klasörde):

    python make_modelfile.py --base qwen2.5:7b --out Modelfile
    ollama create rbf-designer -f Modelfile
"""

import argparse
import glob
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="GGUF icin Modelfile uret")
    p.add_argument("--base", default="qwen2.5:7b",
                   help="sablonu alinacak taban model (once pull edilmis olmali)")
    p.add_argument("--gguf", default=None,
                   help="GGUF dosyasi (varsayilan: klasordeki ilk .gguf)")
    p.add_argument("--out", default="Modelfile")
    p.add_argument("--temperature", default="0.3")
    a = p.parse_args()

    gguf = a.gguf or (sorted(glob.glob("*.gguf")) or [None])[0]
    if not gguf:
        sys.exit("Bu klasorde .gguf bulunamadi — zip'i actiginiz klasorde "
                 "calistirin (icinde rbf-designer...gguf olmali).")

    try:
        cikti = subprocess.run(
            ["ollama", "show", a.base, "--modelfile"],
            capture_output=True, text=True, check=True,
            encoding="utf-8", errors="replace").stdout
    except FileNotFoundError:
        sys.exit("ollama komutu bulunamadi — yeni bir terminal acin veya:\n"
                 "  $env:Path += \";$env:LOCALAPPDATA\\Programs\\Ollama\"")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Taban model okunamadi ({a.base}): {e.stderr.strip()}\n"
                 f"Once cekin: ollama pull {a.base}")

    lines = cikti.splitlines()
    idx = next((i for i, ln in enumerate(lines) if ln.startswith("FROM ")),
               None)
    if idx is None:
        sys.exit("ollama show ciktisinda FROM satiri yok — beklenmedik cikti.")

    # FROM'u yerel GGUF ile degistir; TEMPLATE/PARAMETER vb. aynen tasi.
    govde = [f"FROM ./{gguf}"] + lines[idx + 1:]
    govde.append(f"PARAMETER temperature {a.temperature}")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(govde) + "\n")

    print(f"{a.out} yazildi (taban sablon: {a.base} | gguf: {gguf})")
    if ".Tools" in cikti:
        print("Arac-cagirma sablonu: VAR ({{ .Tools }} bulundu) — "
              "llm_controller calisir.")
    else:
        print("UYARI: taban modelin sablonunda .Tools yok — arac cagirma "
              "calismayabilir. Farkli bir taban deneyin (--base).")
    print(f"Simdi: ollama create rbf-designer -f {a.out}")


if __name__ == "__main__":
    main()
