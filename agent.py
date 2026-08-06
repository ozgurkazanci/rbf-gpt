# -*- coding: utf-8 -*-
"""Cadence tasarım ajanı — iki hattın buluştuğu yer.

Mimari üç katmandır:

  ARAÇLAR (CadenceTools)   : Cadence'a dokunan çağrılar. cadence_bridge
                             üzerinden gerçek Spectre koşar. Bir modelin
                             "elleri" bunlardır; TOOL_SCHEMAS listesi de
                             tam olarak bir LLM'e verilecek araç tanımıdır.

  KONTROLCÜ                : Hangi aracın hangi parametreyle çağrılacağına
                             karar veren beyin. Bugün kurallı (RuleBasedSizer);
                             yarın buraya yerel bir model, sonra sizin
                             eğittiğiniz model takılır — araç katmanı
                             değişmeden.

  KAYIT (history JSON)     : Her adımın (parametre -> ölçüm) çifti diske
                             yazılır. Bu kayıtlar ikisi için de hammaddedir:
                             (a) TurboRBF'i simülatör vekili olarak eğitmek,
                             (b) ileride modeli araç kullanımı üzerinde
                             eğitmek.

Kullanım:

    # Gerçek Cadence ile: eviriciyi Vm = 0.6 V olacak sekilde boyutlandir
    python agent.py size-inverter --target-vm 0.6

    # Cadence olmadan mantik provasi (analitik sahte simulator)
    python agent.py size-inverter --target-vm 0.6 --simulator mock

    # Tek bir olcum
    python agent.py measure --wn 200 --wp 400
"""

import argparse
import json
import math
import time

from cadence_bridge.bridge import CadenceBridge
from cadence_bridge.psf import vtc_metrics

# ---------------------------------------------------------------------------
# ARAÇ TANIMLARI — bir LLM'e verilecek şema budur (henüz model bağlı değil).
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "measure_inverter",
        "description": "CMOS eviriciyi verilen transistor genislikleriyle "
                       "simule eder, DC gecis egrisinden anahtarlama esigi "
                       "(vm), cikis seviyeleri (voh/vol) ve kazanci doner.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wn_nm": {"type": "number",
                          "description": "NMOS genisligi (nanometre)"},
                "wp_nm": {"type": "number",
                          "description": "PMOS genisligi (nanometre)"},
                "l_nm": {"type": "number",
                         "description": "kanal boyu (nm), varsayilan 60"},
            },
            "required": ["wn_nm", "wp_nm"],
        },
    },
    {
        "name": "pdk_info",
        "description": "Kullanilan PDK model klasoru, kose (corner) adi ve "
                       "kose dosyasi yolunu doner.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# ARAÇLAR
# ---------------------------------------------------------------------------

class CadenceTools:
    """Ajanın elleri: gerçek Spectre koşar, ölçüm sözlüğü döndürür."""

    def __init__(self, bridge=None, vdd=1.2, nmos="nch", pmos="pch"):
        self.bridge = bridge or CadenceBridge()
        self.vdd, self.nmos, self.pmos = vdd, nmos, pmos
        self.calls = 0

    def pdk_info(self):
        return self.bridge.discover_models()

    def measure_inverter(self, wn_nm, wp_nm, l_nm=60):
        self.calls += 1
        t0 = time.perf_counter()
        rc, out, err = self.bridge.inverter_vtc(
            wn=f"{wn_nm:g}n", wp=f"{wp_nm:g}n", length=f"{l_nm:g}n",
            vdd=self.vdd, nmos=self.nmos, pmos=self.pmos,
            tag=f"a{self.calls:03d}")
        dt = time.perf_counter() - t0
        if "SPECTRE_RC=0" not in out:
            hata = "\n".join(ln for ln in out.splitlines()
                             if "rror" in ln or "atal" in ln)[:300]
            return {"hata": hata or err or "spectre basarisiz",
                    "wn_nm": wn_nm, "wp_nm": wp_nm, "sure_s": round(dt, 2)}
        raw = out.split("---RAW---", 1)[-1]
        m = vtc_metrics(raw)
        if m.get("hata"):
            return {"hata": m["hata"], "wn_nm": wn_nm, "wp_nm": wp_nm}
        return {"wn_nm": wn_nm, "wp_nm": wp_nm, "l_nm": l_nm,
                "vm": m["vm"], "voh": m["voh"], "vol": m["vol"],
                "gain": round(m["gain"], 2), "nokta": m["n"],
                "sure_s": round(dt, 2)}


class MockTools:
    """Cadence'sız prova: birinci mertebe MOS modeliyle Vm hesaplar.

    Vm = (Vtn + sqrt(r)(Vdd - |Vtp|)) / (1 + sqrt(r)),  r ≈ (µp/µn)(Wp/Wn)
    Kontrol mantığını gerçek simülatör olmadan doğrulamak içindir.
    """

    def __init__(self, vdd=1.2, vtn=0.35, vtp=0.35, mu_ratio=0.4):
        self.vdd, self.vtn, self.vtp, self.mu = vdd, vtn, vtp, mu_ratio
        self.calls = 0

    def pdk_info(self):
        return {"model_dir": "(mock)", "section": "tt", "corner_file": "(mock)"}

    def measure_inverter(self, wn_nm, wp_nm, l_nm=60):
        self.calls += 1
        r = math.sqrt(self.mu * wp_nm / wn_nm)
        vm = (self.vtn + r * (self.vdd - self.vtp)) / (1 + r)
        return {"wn_nm": wn_nm, "wp_nm": wp_nm, "l_nm": l_nm, "vm": vm,
                "voh": self.vdd, "vol": 0.0, "gain": -18.0, "nokta": 121,
                "sure_s": 0.0}


# ---------------------------------------------------------------------------
# KONTROLCÜ — bugün kurallı; modelin takılacağı yer burası.
# ---------------------------------------------------------------------------

class RuleBasedSizer:
    """PMOS/NMOS oranını ayarlayarak hedef Vm'e yakınsar.

    Vm, Wp/Wn oranında monotondur (PMOS güçlendikçe Vm yükselir). Bu yüzden
    önce hedefi kuşatan bir aralık bulunur (oranı 2 kat büyütüp küçülterek),
    sonra ikiye bölme ile daraltılır — türev gerektirmez, gürültüye dayanıklı.
    """

    # Fiziksel olarak makul Wp/Wn araligi. Sinir olmadan arama, hedefe
    # ulasmak icin 400 um genisliginde transistor gibi sacma boyutlar
    # onerebiliyor; ajan bunun yerine "ulasilamaz" demeyi ogrenmeli.
    RATIO_MIN, RATIO_MAX = 0.1, 20.0

    def __init__(self, tools, wn_nm=200.0, l_nm=60.0):
        self.tools = tools
        self.wn, self.l = wn_nm, l_nm
        self.history = []
        self.durum = "calisiyor"

    def _measure(self, ratio, note):
        res = self.tools.measure_inverter(self.wn, self.wn * ratio, self.l)
        res.update(ratio=round(ratio, 4), adim=len(self.history) + 1,
                   not_=note)
        self.history.append(res)
        vm = res.get("vm")
        print(f"  adim {res['adim']:2d} | Wp/Wn={ratio:6.3f} | "
              + (f"Vm={vm:.4f} V" if vm is not None
                 else f"HATA: {res.get('hata', '?')}")
              + f" | {res.get('sure_s', 0)}s | {note}")
        return vm

    def solve(self, target_vm, tol=0.005, max_iters=16, ratio0=2.0):
        print(f"\nHedef: Vm = {target_vm} V (tolerans +/-{tol} V)")
        vm0 = self._measure(ratio0, "baslangic")
        if vm0 is None:
            return None
        if abs(vm0 - target_vm) <= tol:
            return self.history[-1]

        # 1) Kuşatma: hedefin diğer tarafına düşene kadar oranı ikiye katla/böl
        lo = hi = None
        ratio, vm = ratio0, vm0
        for _ in range(10):
            if len(self.history) >= max_iters:
                break
            if vm < target_vm:          # PMOS'u güçlendir -> Vm yükselir
                lo = ratio
                ratio *= 2.0
            else:
                hi = ratio
                ratio /= 2.0
            if not (self.RATIO_MIN <= ratio <= self.RATIO_MAX):
                self.durum = "sinirda"
                print(f"  ! Wp/Wn={ratio:.3g} fiziksel sinirlarin disinda "
                      f"({self.RATIO_MIN}-{self.RATIO_MAX}); hedef bu "
                      f"topoloji/kanal boyu ile ulasilamiyor")
                break
            vm = self._measure(ratio, "kusatma")
            if vm is None:
                return None
            if abs(vm - target_vm) <= tol:
                return self.history[-1]
            if (lo is not None and vm > target_vm) or \
               (hi is not None and vm < target_vm):
                if vm > target_vm:
                    hi = ratio
                else:
                    lo = ratio
                break
        if lo is None or hi is None:
            self.durum = "kusatilamadi"
            print("  ! hedef kusatilamadi — kanal boyu (--l) veya topoloji "
                  "degistirilmeli")
            return self._en_iyi()

        # 2) Daraltma: logaritmik ikiye bölme (oran çarpımsal bir büyüklük)
        while len(self.history) < max_iters:
            ratio = math.exp(0.5 * (math.log(lo) + math.log(hi)))
            vm = self._measure(ratio, "daraltma")
            if vm is None:
                return None
            if abs(vm - target_vm) <= tol:
                return self.history[-1]
            if vm < target_vm:
                lo = ratio
            else:
                hi = ratio
        self.durum = "iterasyon_bitti"
        return self._en_iyi(target_vm)

    def _en_iyi(self, target_vm=None):
        olculenler = [h for h in self.history if h.get("vm") is not None]
        if not olculenler:
            return None
        if target_vm is None:
            return olculenler[-1]
        return min(olculenler, key=lambda h: abs(h["vm"] - target_vm))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _tools(args):
    if args.simulator == "mock":
        print("Simulator: MOCK (analitik model — Cadence calistirilmiyor)")
        return MockTools(vdd=args.vdd)
    print("Simulator: Spectre (gercek PDK)")
    t = CadenceTools(vdd=args.vdd)
    try:
        info = t.pdk_info()
    except RuntimeError as exc:
        raise SystemExit(
            f"\nCadence'a ulasilamadi: {exc}\n"
            "Bu komut Windows PowerShell'den (veya WSL icinden) "
            "calistirilmali.\nMantigi Cadence'siz denemek icin: "
            "--simulator mock")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if info.get("hata") or not info.get("section"):
        raise SystemExit("PDK kosesi bulunamadi — once "
                         "'python -m cadence_bridge.demo pdk' calistirin.")
    return t


def main():
    # Ortak secenekler hem alt komuttan once hem sonra yazilabilsin diye
    # parents ile her iki seviyeye de eklenir.
    # SUPPRESS sart: varsayilan deger yazilsaydi, alt komut ayristiricisi
    # komuttan ONCE verilen degeri ezerdi.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--simulator", choices=["spectre", "mock"],
                        default=argparse.SUPPRESS)
    common.add_argument("--vdd", type=float, default=argparse.SUPPRESS)
    common.add_argument("--history", default=argparse.SUPPRESS,
                        help="adim kayitlarinin yazilacagi JSON dosyasi")

    p = argparse.ArgumentParser(description="Cadence tasarim ajani",
                                parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("size-inverter", parents=[common],
                        help="hedef Vm icin PMOS/NMOS oranini bul")
    sp.add_argument("--target-vm", type=float, default=0.6)
    sp.add_argument("--tol", type=float, default=0.005)
    sp.add_argument("--max-iters", type=int, default=16)
    sp.add_argument("--wn", type=float, default=200.0, help="NMOS W (nm)")
    sp.add_argument("--l", type=float, default=60.0, help="kanal boyu (nm)")

    sp = sub.add_parser("measure", parents=[common], help="tek bir olcum")
    sp.add_argument("--wn", type=float, default=200.0)
    sp.add_argument("--wp", type=float, default=400.0)
    sp.add_argument("--l", type=float, default=60.0)

    args = p.parse_args()
    for key, default in (("simulator", "spectre"), ("vdd", 1.2),
                         ("history", "agent_history.json")):
        setattr(args, key, getattr(args, key, default))
    tools = _tools(args)

    if args.cmd == "measure":
        res = tools.measure_inverter(args.wn, args.wp, args.l)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    sizer = RuleBasedSizer(tools, wn_nm=args.wn, l_nm=args.l)
    best = sizer.solve(args.target_vm, tol=args.tol, max_iters=args.max_iters)

    with open(args.history, "w", encoding="utf-8") as f:
        json.dump({"hedef_vm": args.target_vm, "simulator": args.simulator,
                   "durum": sizer.durum, "adimlar": sizer.history},
                  f, indent=2, ensure_ascii=False)

    print(f"\n{len(sizer.history)} simulasyon | kayit: {args.history}")
    if best and best.get("vm") is not None:
        print(f"SONUC: Wp/Wn = {best['ratio']} "
              f"(Wn={best['wn_nm']:g}nm, Wp={best['wp_nm']:g}nm) "
              f"-> Vm = {best['vm']:.4f} V")
        if abs(best["vm"] - args.target_vm) <= args.tol:
            print("AJAN BASARILI: hedef tolerans icinde yakalandi.")
        elif sizer.durum in ("sinirda", "kusatilamadi"):
            print(f"AJAN DURDU ({sizer.durum}): hedef, makul transistor "
                  f"boyutlariyla bu topolojide ulasilamiyor — en yakin "
                  f"sonuc yukarida.")
        else:
            print("Hedefe tam ulasilamadi — --max-iters artirilabilir.")
    else:
        print("SONUC: olcum alinamadi.")


if __name__ == "__main__":
    main()
