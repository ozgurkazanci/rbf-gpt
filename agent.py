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
import random
import time

from cadence_bridge.bridge import CadenceBridge
from cadence_bridge.psf import vtc_metrics

# ---------------------------------------------------------------------------
# ARAÇ TANIMLARI — bir LLM'e verilecek şema budur (henüz model bağlı değil).
# ---------------------------------------------------------------------------

# TSMC65 cekirdek cihazlari icin asgari cizim genisligi. Bunun altindaki
# W degerlerinde Spectre "Error found during initial setup" verir; ajan
# tasarim kuralini bilmezse simulasyonlarin bir kismini bosa harcar.
W_MIN_NM = 120.0
L_MIN_NM = 60.0

TOOL_SCHEMAS = [
    {
        "name": "measure_inverter",
        "description": "CMOS eviriciyi verilen transistor genislikleriyle "
                       "simule eder. Her arka ucta 'vm' (anahtarlama esigi, "
                       "V) ve 'gain' doner; 'voh'/'vol' yalnizca gercek "
                       "simulasyon (spectre) ve mock arka uclarinda bulunur, "
                       "vekil (surrogate) arka ucu donmez. Genislik/boy "
                       "tasarim kurallarina uymayan cagrilar 'hata' alani "
                       "ile reddedilir.",
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
        self.wn = max(wn_nm, W_MIN_NM)      # tasarim kurali
        self.l = max(l_nm, L_MIN_NM)
        # Oranin alt siniri, Wp'nin de kurala uymasini garanti eder.
        self.ratio_min = max(self.RATIO_MIN, W_MIN_NM / self.wn)
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
            if not (self.ratio_min <= ratio <= self.RATIO_MAX):
                # Pes etmeden once SINIRDA bir olcum al: hedef, son olculen
                # nokta ile sinir arasindaki bantta olabilir.
                sinir = min(max(ratio, self.ratio_min), self.RATIO_MAX)
                vm = self._measure(sinir, "sinir")
                if vm is None:
                    return None
                if abs(vm - target_vm) <= tol:
                    return self.history[-1]
                if vm < target_vm:
                    lo = sinir
                else:
                    hi = sinir
                if lo is not None and hi is not None:
                    break                      # sinir olcumu hedefi kusatti
                self.durum = "sinirda"
                print(f"  ! sinirda da (Wp/Wn={sinir:.3g}, "
                      f"{self.ratio_min:.3g}-{self.RATIO_MAX} araligi; "
                      f"W_min={W_MIN_NM:g} nm kurali dahil) hedef ayni "
                      f"tarafta kaldi — bu topoloji/kanal boyu ile "
                      f"ulasilamiyor")
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
# VERİ TOPLAMA — vekil modelin (surrogate) eğitim kümesi
# ---------------------------------------------------------------------------

def collect_dataset(tools, n=40, wn_min=W_MIN_NM, wn_max=600.0,
                    ratio_min=0.3, ratio_max=8.0, l_nm=60.0, seed=0):
    """Tasarım uzayından log-düzgün örnekler alıp gerçek simülasyon koşar.

    Genişlikler çarpımsal büyüklükler olduğu için örnekleme log ölçekte
    yapılır: 100-600 nm aralığında düzgün örnekleme küçük genişlikleri
    yeterince temsil etmezdi.
    """
    rng = random.Random(seed)
    veri = []
    wn_min = max(wn_min, W_MIN_NM)          # tasarim kurali: W >= 120 nm
    l_nm = max(l_nm, L_MIN_NM)
    print(f"{n} nokta toplanacak (Wn {wn_min:g}-{wn_max:g} nm, "
          f"Wp/Wn {ratio_min:g}-{ratio_max:g}, W_min={W_MIN_NM:g} nm)")
    for i in range(n):
        wn = math.exp(rng.uniform(math.log(wn_min), math.log(wn_max)))
        ratio = math.exp(rng.uniform(math.log(ratio_min), math.log(ratio_max)))
        # Wp da kurala uymali; ihlal eden ornekler simulasyonu bosa harcar.
        wp = max(wn * ratio, W_MIN_NM)
        ratio = wp / wn
        res = tools.measure_inverter(round(wn, 1), round(wp, 1), l_nm)
        res["ratio"] = round(ratio, 4)
        veri.append(res)
        vm = res.get("vm")
        print(f"  {i+1:3d}/{n} | Wn={res['wn_nm']:7.1f} Wp={res['wp_nm']:8.1f} | "
              + (f"Vm={vm:.4f} V" if vm is not None
                 else f"HATA: {res.get('hata', '?')[:60]}")
              + f" | {res.get('sure_s', 0)}s")
    basarili = [v for v in veri if v.get("vm") is not None]
    print(f"\n{len(basarili)}/{n} nokta basarili")
    return veri


# ---------------------------------------------------------------------------
# TOPLU TARAMA — vekilin asıl gücü: binlerce adayı saniyede elemek
# ---------------------------------------------------------------------------

def screen_designs(surrogate, target_vm, n=10000, wn_min=W_MIN_NM,
                   wn_max=600.0, ratio_min=0.3, ratio_max=8.0, l_nm=60.0,
                   top=5, seed=0):
    """Tasarım uzayını vekille tarar, hedefe en yakın adayları döndürür.

    Kaba kuvvetle n gerçek simülasyon saatler sürerdi; vekil aynı taramayı
    tek matris çarpımıyla yapar. Dönen adaylar 'en iyi tahmin' listesidir —
    son sözü yine gerçek Spectre söylemelidir (--verify).
    """
    rng = random.Random(seed)
    wn_min = max(wn_min, W_MIN_NM)
    l_kul = max(l_nm, L_MIN_NM)                 # tasarim kurali: L >= 60 nm
    ciftler = []
    for _ in range(n):
        wn = math.exp(rng.uniform(math.log(wn_min), math.log(wn_max)))
        wp = max(wn * math.exp(rng.uniform(math.log(ratio_min),
                                           math.log(ratio_max))), W_MIN_NM)
        ciftler.append((round(wn, 1), round(wp, 1)))
    # W_MIN kelepcesi ayni (wn, wp) ciftini defalarca uretir; kopyalar hem
    # top-k'yi ayni tasarimla doldurur hem de --verify'da ozdes Spectre
    # kosularini bosa harcar — tekillestir.
    ciftler = list(dict.fromkeys(ciftler))
    wns = [c[0] for c in ciftler]
    wps = [c[1] for c in ciftler]

    t0 = time.perf_counter()
    tahmin = surrogate.predict_many(wns, wps, l_kul)
    dt = time.perf_counter() - t0

    adaylar = sorted(
        ({"wn_nm": wns[i], "wp_nm": wps[i], "l_nm": l_kul,
          "ratio": round(wps[i] / wns[i], 4), "vm": tahmin["vm"][i],
          **({"gain": tahmin["gain"][i]} if "gain" in tahmin else {})}
         for i in range(len(wns))),
        key=lambda a: abs(a["vm"] - target_vm))[:top]
    return adaylar, dt, len(wns)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _append_history(path, kayitlar, kaynak="spectre"):
    """Ölçümleri mevcut geçmiş dosyasına ekler (yoksa oluşturur).

    Dosya, surrogate.load_dataset'in okuduğu {"adimlar": [...]} biçimini
    korur; böylece her gerçek ölçüm ileride vekil eğitimine girebilir.
    """
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d.get("adimlar"), list):
            d = {"adimlar": []}
    except (OSError, json.JSONDecodeError):
        d = {"adimlar": []}
    d.setdefault("kaynak", kaynak)
    d["adimlar"].extend(kayitlar)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def _tools(args):
    if args.simulator == "mock":
        print("Simulator: MOCK (analitik model — Cadence calistirilmiyor)")
        return MockTools(vdd=args.vdd)
    if args.simulator == "surrogate":
        from surrogate import SurrogateTools      # torch sadece burada gerekir
        t = SurrogateTools(args.model)
        print(f"Simulator: VEKIL TurboRBF ({args.model}) — Spectre "
              f"calistirilmiyor")
        for k, v in t.pdk_info().items():
            print(f"  {k}: {v}")
        return t
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
    common.add_argument("--simulator", choices=["spectre", "mock", "surrogate"],
                        default=argparse.SUPPRESS)
    common.add_argument("--vdd", type=float, default=argparse.SUPPRESS)
    common.add_argument("--model", default=argparse.SUPPRESS,
                        help="vekil model dosyasi (--simulator surrogate)")
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
    sp.add_argument("--verify", action="store_true",
                    help="vekille bulunan sonucu gercek Spectre ile dogrula")

    sp = sub.add_parser("screen", parents=[common],
                        help="vekille binlerce adayi tara, en iyileri sec")
    sp.add_argument("--target-vm", type=float, default=0.6)
    sp.add_argument("--n", type=int, default=10000)
    sp.add_argument("--top", type=int, default=5)
    sp.add_argument("--wn-min", type=float, default=W_MIN_NM)
    sp.add_argument("--wn-max", type=float, default=600.0)
    sp.add_argument("--ratio-min", type=float, default=0.3)
    sp.add_argument("--ratio-max", type=float, default=8.0)
    sp.add_argument("--l", type=float, default=60.0)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--verify", action="store_true",
                    help="en iyi adaylari gercek Spectre ile dogrula")

    sp = sub.add_parser("collect", parents=[common],
                        help="tasarim uzayindan veri topla (vekil egitimi icin)")
    sp.add_argument("--n", type=int, default=40)
    sp.add_argument("--wn-min", type=float, default=100.0)
    sp.add_argument("--wn-max", type=float, default=600.0)
    sp.add_argument("--ratio-min", type=float, default=0.3)
    sp.add_argument("--ratio-max", type=float, default=8.0)
    sp.add_argument("--l", type=float, default=60.0)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default="dataset.json")

    sp = sub.add_parser("measure", parents=[common], help="tek bir olcum")
    sp.add_argument("--wn", type=float, default=200.0)
    sp.add_argument("--wp", type=float, default=400.0)
    sp.add_argument("--l", type=float, default=60.0)

    args = p.parse_args()
    for key, default in (("simulator", "spectre"), ("vdd", 1.2),
                         ("model", "surrogate.pt"),
                         ("history", "agent_history.json")):
        setattr(args, key, getattr(args, key, default))
    tools = _tools(args)

    if args.cmd == "measure":
        res = tools.measure_inverter(args.wn, args.wp, args.l)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        _append_history(args.history, [res], kaynak=args.simulator)
        print(f"kayit: {args.history}")
        return

    if args.cmd == "screen":
        if not hasattr(tools, "predict_many"):
            raise SystemExit("screen komutu vekil ister: --simulator "
                             "surrogate (once surrogate.py train).")
        adaylar, dt, tekil = screen_designs(
            tools, args.target_vm, n=args.n, wn_min=args.wn_min,
            wn_max=args.wn_max, ratio_min=args.ratio_min,
            ratio_max=args.ratio_max, l_nm=args.l, top=args.top,
            seed=args.seed)
        print(f"\n{args.n} aday ({tekil} tekil) {dt:.2f} saniyede tarandi "
              f"(Spectre ile ~{tekil * 1.3 / 3600:.1f} saat surerdi)")
        print(f"Hedef Vm = {args.target_vm} V | en iyi {len(adaylar)} aday:")
        for i, a in enumerate(adaylar, 1):
            print(f"  {i}. Wn={a['wn_nm']:6.1f} Wp={a['wp_nm']:7.1f} "
                  f"(oran {a['ratio']:.3f}) | tahmini Vm={a['vm']:.4f} V")
        gercekler = []
        if args.verify:
            print("\nGercek Spectre dogrulamasi:")
            gercek_arac = CadenceTools(vdd=args.vdd)
            for i, a in enumerate(adaylar, 1):
                # Vekil hangi L'de tahmin ettiyse dogrulama da o L'de kosar
                g = gercek_arac.measure_inverter(a["wn_nm"], a["wp_nm"],
                                                 a["l_nm"])
                if g.get("vm") is None:
                    print(f"  {i}. HATA: {g.get('hata', '?')[:70]}")
                    continue
                gercekler.append(g)
                fark = (g["vm"] - a["vm"]) * 1000
                print(f"  {i}. gercek Vm={g['vm']:.4f} V | vekil sapmasi "
                      f"{fark:+.1f} mV | hedefe uzaklik "
                      f"{abs(g['vm'] - args.target_vm)*1000:.1f} mV")
        # Gercek olcumler vekilin gelecekteki egitim verisidir — kaydet.
        if gercekler:
            _append_history(args.history, gercekler, kaynak="spectre")
            print(f"\n{len(gercekler)} gercek olcum kaydedildi: "
                  f"{args.history} (surrogate.py train --data ile "
                  f"yeniden egitimde kullanilabilir)")
        return

    if args.cmd == "collect":
        veri = collect_dataset(tools, n=args.n, wn_min=args.wn_min,
                               wn_max=args.wn_max, ratio_min=args.ratio_min,
                               ratio_max=args.ratio_max, l_nm=args.l,
                               seed=args.seed)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"kaynak": args.simulator, "adimlar": veri}, f,
                      indent=2, ensure_ascii=False)
        print(f"Kayit: {args.out}")
        print(f"Sonraki adim: python surrogate.py train --data {args.out}")
        return

    sizer = RuleBasedSizer(tools, wn_nm=args.wn, l_nm=args.l)
    best = sizer.solve(args.target_vm, tol=args.tol, max_iters=args.max_iters)

    print(f"\n{len(sizer.history)} simulasyon")
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

        # Hibrit akis: vekil hizli tarar, gercek simulator son sozu soyler.
        if getattr(args, "verify", False) and args.simulator == "surrogate":
            print("\nVekilin buldugu nokta gercek Spectre ile dogrulaniyor...")
            gercek = CadenceTools(vdd=args.vdd).measure_inverter(
                best["wn_nm"], best["wp_nm"], best.get("l_nm", args.l))
            if gercek.get("vm") is None:
                print(f"  dogrulama basarisiz: {gercek.get('hata')}")
            else:
                # Gercek olcum da gecmise girer — vekilin egitim verisidir.
                gercek["not_"] = "dogrulama"
                sizer.history.append(gercek)
                fark = gercek["vm"] - best["vm"]
                print(f"  vekil: {best['vm']:.4f} V | gercek: "
                      f"{gercek['vm']:.4f} V | fark: {fark*1000:+.1f} mV")
                if abs(gercek["vm"] - args.target_vm) <= args.tol:
                    print("  DOGRULANDI: gercek simulasyon da hedef icinde.")
                else:
                    print("  Vekil sapmis — bu nokta gecmis dosyasina "
                          "kaydedildi; surrogate.py train ile yeniden "
                          "egitim isabeti artirir.")
    else:
        print("SONUC: olcum alinamadi.")

    with open(args.history, "w", encoding="utf-8") as f:
        json.dump({"hedef_vm": args.target_vm, "simulator": args.simulator,
                   "durum": sizer.durum, "adimlar": sizer.history},
                  f, indent=2, ensure_ascii=False)
    print(f"kayit: {args.history}")


if __name__ == "__main__":
    main()
