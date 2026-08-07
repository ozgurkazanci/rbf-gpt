# -*- coding: utf-8 -*-
"""LLM kontrolcüsü — kurallı beynin yerine dil modeli.

agent.py'deki RuleBasedSizer sabit bir görevi çözer; buradaki kontrolcü ise
SERBEST METİN görev alır ("eviriciyi Vm 0.55V olacak sekilde boyutlandir,
Wp 1um'i gecmesin") ve karar döngüsünü bir dil modeline bırakır:

    model konuşur -> araç çağırır -> ölçüm döner -> model değerlendirir -> ...

Araçlar agent.py'dekilerle AYNIDIR (CadenceTools/SurrogateTools/MockTools);
model yalnızca hangi aracı hangi parametreyle çağıracağını seçer. Yani bu
dosya, "kendi modelim Cadence kullanacak" hedefinin ilk çalışan hâlidir —
bugün Ollama'daki açık bir model, yarın sizin eğittiğiniz model.

Kurulum (Windows):
    winget install Ollama.Ollama
    ollama pull qwen2.5:7b          # arac cagirmayi bilen yerel model

Kullanım:
    python llm_controller.py --task "Eviriciyi Vm=0.6V olacak sekilde
        boyutlandir" --simulator surrogate
    python llm_controller.py --task "..." --simulator spectre   # gercek PDK
"""

import argparse
import http.client
import json
import urllib.error
import urllib.request

from agent import (TOOL_SCHEMAS, CadenceTools, MockTools, L_MIN_NM,
                   W_MIN_NM)

SISTEM_TALIMATI = f"""Sen bir analog devre tasarim ajanisin. TSMC65 surecinde
CMOS evirici tasarimi yapiyorsun. Elindeki araclarla olcum alabilirsin.

Kurallar:
- Transistor genislikleri en az {W_MIN_NM:g} nm olmalidir (tasarim kurali).
- Az sayida olcumle hedefe ulasmaya calis; her olcumden sonra sonucu
  degerlendirip bir sonraki denemeni ona gore sec. Vm, Wp/Wn oraniyla
  birlikte artar.
- Hedefe ulastiginda (veya ulasamayacagini anladiginda) arac cagirmayi
  birak ve sonucu tek paragrafta ozetle: bulunan Wn/Wp, olculen degerler
  ve hedefle karsilastirma."""


class OllamaLLM:
    """Yerel Ollama sunucusuyla araç-çağrılı sohbet."""

    def __init__(self, model="qwen2.5:7b", host="http://localhost:11434"):
        self.model, self.host = model, host

    def chat(self, messages, tools):
        # Anthropic tarzi input_schema -> Ollama/OpenAI tarzi parameters
        ollama_tools = [{"type": "function",
                         "function": {"name": t["name"],
                                      "description": t["description"],
                                      "parameters": t["input_schema"]}}
                        for t in tools]
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps({"model": self.model, "messages": messages,
                             "tools": ollama_tools, "stream": False}
                            ).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                govde = r.read()
        except urllib.error.HTTPError as exc:
            # Ollama'nin JSON hata govdesi asil teshisi tasir (orn. model
            # cekilmemis) — yutma, kullaniciya goster.
            try:
                detay = exc.read().decode("utf-8", "replace")[:300]
            except OSError:
                detay = ""
            raise RuntimeError(
                f"Ollama HTTP {exc.code}: {detay or exc.reason}\n"
                f"Model cekilmemis olabilir: ollama pull {self.model}")
        except (OSError, http.client.HTTPException) as exc:
            # URLError, TimeoutError, ConnectionReset, IncompleteRead...
            # baglanti VE okuma asamasi hatalarinin tamami burada.
            raise RuntimeError(
                f"Ollama'ya ulasilamadi ({self.host}): {exc}\n"
                "Kurulum: winget install Ollama.Ollama\n"
                f"Model:   ollama pull {self.model}")
        try:
            yanit = json.loads(govde)
        except json.JSONDecodeError:
            raise RuntimeError(f"Ollama gecersiz JSON dondurdu: "
                               f"{govde[:200]!r}")
        msg = yanit.get("message")
        if not isinstance(msg, dict):
            raise RuntimeError(f"Beklenmedik Ollama yaniti: "
                               f"{str(yanit)[:200]}")
        return msg


class LLMAgent:
    """Araç-çağrı döngüsü: model karar verir, araçlar ölçer."""

    def __init__(self, llm, tools, max_turns=16, max_measurements=24):
        self.llm, self.tools, self.max_turns = llm, tools, max_turns
        # max_turns tur sayisini sinirlar ama TEK turda 50 arac cagrisi
        # gelebilir; gercek Spectre'da her biri dakikalara mal olur. Olcum
        # butcesi bunu kesin olarak sinirlar.
        self.max_measurements = max_measurements
        self.olcum_sayisi = 0

    def _call_tool(self, name, args):
        try:
            if name == "pdk_info":
                return self.tools.pdk_info()
            if name == "measure_inverter":
                if self.olcum_sayisi >= self.max_measurements:
                    return {"hata": f"olcum butcesi doldu "
                                    f"({self.max_measurements} olcum); yeni "
                                    f"olcum alma, eldeki verilerle sonuca "
                                    f"git ve ozetle"}
                wn = float(args["wn_nm"])
                wp = float(args["wp_nm"])
                l = float(args.get("l_nm", 60))
                # Tasarim kurali her arka ucta zorlanir: vekil/mock aksi
                # halde uretilebilir olmayan boyutlara makul degerler
                # dondurup modeli yanlis odullendirirdi.
                if wn < W_MIN_NM or wp < W_MIN_NM or l < L_MIN_NM:
                    return {"hata": f"tasarim kurali ihlali: genislikler >= "
                                    f"{W_MIN_NM:g} nm, kanal boyu >= "
                                    f"{L_MIN_NM:g} nm olmali "
                                    f"(istenen: wn={wn:g}, wp={wp:g}, "
                                    f"l={l:g})"}
                self.olcum_sayisi += 1
                return self.tools.measure_inverter(wn, wp, l)
            return {"hata": f"bilinmeyen arac: {name}"}
        except Exception as exc:                      # modelin hatasi ona doner
            return {"hata": f"{type(exc).__name__}: {exc}"}

    def run(self, task, verbose=True):
        messages = [{"role": "system", "content": SISTEM_TALIMATI},
                    {"role": "user", "content": task}]
        for turn in range(1, self.max_turns + 1):
            msg = self.llm.chat(messages, TOOL_SCHEMAS)
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                if verbose:
                    print(f"\n=== MODELIN SONUCU (tur {turn}, "
                          f"{self.olcum_sayisi} olcum) ===")
                    print(msg.get("content", "(bos)"))
                return msg.get("content", ""), messages
            for c in calls:
                fn = c.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments") or {}
                if isinstance(args, str):             # bazi modeller str doner
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                sonuc = self._call_tool(name, args)
                if verbose:
                    kisa = {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in sonuc.items() if k != "vekil"}
                    print(f"  tur {turn:2d} | {name}({args}) -> {kisa}")
                messages.append({"role": "tool", "name": name,
                                 "content": json.dumps(sonuc)})
        if verbose:
            print(f"\n! {self.max_turns} turda sonuclanmadi.")
        return None, messages


def make_tools(simulator, model_path="surrogate.pt", vdd=1.2):
    if simulator == "mock":
        return MockTools(vdd=vdd)
    if simulator == "surrogate":
        from surrogate import SurrogateTools
        return SurrogateTools(model_path)
    return CadenceTools(vdd=vdd)


def main():
    p = argparse.ArgumentParser(description="LLM'li Cadence tasarim ajani")
    p.add_argument("--task", required=True,
                   help='gorev, serbest metin (ornek: "Eviriciyi Vm=0.6V '
                        'olacak sekilde boyutlandir")')
    p.add_argument("--simulator", choices=["spectre", "surrogate", "mock"],
                   default="surrogate")
    p.add_argument("--model", default="surrogate.pt",
                   help="vekil model dosyasi (--simulator surrogate)")
    p.add_argument("--llm", default="qwen2.5:7b",
                   help="Ollama model adi (qwen2.5:7b, llama3.1:8b ...)")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument("--max-measurements", type=int, default=24,
                   help="toplam olcum butcesi (gercek Spectre'da maliyeti "
                        "sinirlar)")
    p.add_argument("--vdd", type=float, default=1.2)
    args = p.parse_args()

    tools = make_tools(args.simulator, args.model, args.vdd)
    print(f"Simulator: {args.simulator} | LLM: {args.llm} | "
          f"olcum butcesi: {args.max_measurements}")
    print(f"Gorev: {args.task}\n")
    agent = LLMAgent(OllamaLLM(args.llm, args.host), tools,
                     max_turns=args.max_turns,
                     max_measurements=args.max_measurements)
    try:
        agent.run(args.task)
    except RuntimeError as exc:
        raise SystemExit(f"\n{exc}")


if __name__ == "__main__":
    main()
