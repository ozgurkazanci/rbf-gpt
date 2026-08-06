# -*- coding: utf-8 -*-
"""Cadence köprüsü CLI'ı — köprüyü elle test etmek için.

Kullanım (Windows PowerShell'den veya WSL içinden, repo kökünde):

    python -m cadence_bridge.demo check                 # araçlar görünüyor mu?
    python -m cadence_bridge.demo virtuoso              # Virtuoso GUI'yi başlat
    python -m cadence_bridge.demo skill betik.il        # SKILL betiği koş (batch)
    python -m cadence_bridge.demo ocean betik.ocn       # OCEAN betiği koş (batch)
    python -m cadence_bridge.demo spectre devre.scs     # Spectre netlist koş
    python -m cadence_bridge.demo tcl genus akis.tcl    # dijital araç + TCL
    python -m cadence_bridge.demo sh "ls -la"           # WSL'de serbest komut

    --distro Alma_EDA --workdir /home/tonxiao/work65    # varsayılanlar bunlar
"""

import argparse

from .bridge import (CadenceBridge, DEFAULT_DISTRO, DEFAULT_USER,
                     DEFAULT_WORKDIR)


def main():
    p = argparse.ArgumentParser(description="Cadence WSL koprusu")
    p.add_argument("--distro", default=DEFAULT_DISTRO)
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    p.add_argument("--user", default=DEFAULT_USER,
                   help="WSL kullanicisi (Cadence ayarlari bu kullanicinin "
                        ".bashrc'sinden yuklenir)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="WSL erisimi + arac PATH kontrolu")
    sub.add_parser("env", help="ortam teshisi: PATH, bashrc, kurulum dizini")
    sub.add_parser("firstsim", help="uctan uca ilk Spectre simulasyonu (RC devresi)")
    sub.add_parser("pdk", help="PDK envanteri: kitler, spectre modelleri, koseler")
    sub.add_parser("invsim", help="TSMC65 modelleriyle evirici VTC simulasyonu")
    sub.add_parser("virtuoso", help="Virtuoso GUI'yi baslat (virtuoso -64 &)")
    for name, hlp in [("skill", "SKILL betigi (.il) batch calistir"),
                      ("ocean", "OCEAN betigi (.ocn) batch calistir"),
                      ("spectre", "Spectre netlist (.scs) calistir")]:
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument("script")
    sp = sub.add_parser("tcl", help="dijital arac + TCL betigi (batch)")
    sp.add_argument("tool", help="genus / innovus / modus ...")
    sp.add_argument("script")
    sp = sub.add_parser("sh", help="WSL icinde serbest komut")
    sp.add_argument("command")
    sp = sub.add_parser("lib", help="Cadence kutuphanesinin hucrelerini listele")
    sp.add_argument("library", help="kutuphane adi (workdir altinda) veya tam yol")

    args, extra = p.parse_known_args()
    if extra:
        # "virtuoso -64 &" aliskanligina hosgoru: bayraklar zaten kopru
        # icinde uygulanir, fazlaliklari hatayla kesmek yerine yok sayariz.
        print(f"(not: fazladan argumanlar yok sayildi: {' '.join(extra)})")
    br = CadenceBridge(distro=args.distro, workdir=args.workdir,
                       user=args.user)

    if args.cmd == "check":
        for key, val in br.check().items():
            print(f"{key:12s}: {val}")
        return

    if args.cmd == "env":
        for key, val in br.env_report().items():
            print(f"===== {key} =====")
            print(val)
            print()
        return

    if args.cmd == "pdk":
        for key, val in br.pdk_report().items():
            print(f"===== {key} =====")
            print(val)
            print()
        return

    if args.cmd == "invsim":
        import re
        info, rc, out, err = br.inverter_sim()
        for k, v in info.items():
            print(f"{k:12s}: {v}")
        if "---RAW---" in out:
            head, raw = out.split("---RAW---", 1)
            print(head.strip())
        else:
            raw = out
            print(out)
        if err:
            print("--- stderr ---")
            print(err)
        # vin ve vout dosyadan okunur (sweep degerleri varsayilmaz)
        outs = [float(x) for x in re.findall(r'"out"\s+([-+0-9.eE]+)', raw)]
        vin = [float(x) for x in re.findall(r'"in"\s+([-+0-9.eE]+)', raw)]
        if len(outs) < 10 or len(vin) < 10:
            print("\nSONUC: VTC verisi ayristirilamadi — ciktiyi yapistirin,"
                  " birlikte bakalim.")
            return
        n = min(len(vin), len(outs))
        vin, outs = vin[:n], outs[:n]
        vdd = float(info.get("vdd", 1.2))
        # anahtarlama esigi: out'un vin'i kestigi nokta (out ~= vin)
        vm = None
        for i in range(1, n):
            if (outs[i - 1] - vin[i - 1]) * (outs[i] - vin[i]) <= 0:
                # dogrusal interpolasyon
                d0 = outs[i - 1] - vin[i - 1]
                d1 = outs[i] - vin[i]
                t = d0 / (d0 - d1) if d0 != d1 else 0.5
                vm = vin[i - 1] + t * (vin[i] - vin[i - 1])
                break
        print(f"\nSONUC: {n} noktali VTC alindi | Vin: {vin[0]:.3f} -> "
              f"{vin[-1]:.3f} V | V(out): {outs[0]:.3f} V -> {outs[-1]:.3f} V")
        if vm is not None:
            print(f"Anahtarlama esigi Vm = {vm:.3f} V "
                  f"(VDD/2 = {vdd/2:.2f} V civari beklenir)")
        if outs[0] > 0.9 * vdd and outs[-1] < 0.1 * vdd:
            print("GERCEK PDK DOGRULANDI: TSMC65 transistorleriyle evirici"
                  " karakteristigi dogru cikti.")
        else:
            print("DIKKAT: VTC beklenen sekle uymuyor — ciktiyla birlikte"
                  " degerlendirelim.")
        return

    if args.cmd == "firstsim":
        import re
        rc, out, err = br.first_sim()
        print(out)
        if err:
            print("--- stderr ---")
            print(err)
        vals = re.findall(r'"out"\s+([-+0-9.eE]+)', out)
        if not vals:
            print("\nSONUC: V(out) verisi ayristirilamadi — yukaridaki ciktiyi"
                  " yapistirin, birlikte bakalim.")
            return
        v = float(vals[-1])
        print(f"\nSONUC: V(out) son deger = {v:.4f} V (beklenen ~1.0 V)")
        if 0.95 <= v <= 1.05:
            print("KOPRU DOGRULANDI: netlist yaz -> simule et -> oku dongusu"
                  " uctan uca calisiyor.")
        else:
            print("DIKKAT: deger beklenenden sapmis — cikti ile birlikte"
                  " degerlendirelim.")
        return

    if args.cmd == "virtuoso":
        rc, out, err = br.launch_virtuoso()
        print(out or err or f"rc={rc}")
        print("Virtuoso arka planda basladi (WSLg penceresi acilmali).")
        return

    runner = {"skill": br.run_skill, "ocean": br.run_ocean,
              "spectre": br.run_spectre}
    if args.cmd in runner:
        rc, out, err = runner[args.cmd](args.script)
    elif args.cmd == "tcl":
        rc, out, err = br.run_tcl(args.tool, args.script)
    elif args.cmd == "lib":
        rc, out, err = br.list_library(args.library)
    else:  # sh
        rc, out, err = br.run(args.command)

    print(f"--- rc={rc} ---")
    if out:
        print(out)
    if err:
        print("--- stderr ---")
        print(err)


if __name__ == "__main__":
    main()
