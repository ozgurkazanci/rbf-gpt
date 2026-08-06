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

from .bridge import CadenceBridge, DEFAULT_DISTRO, DEFAULT_WORKDIR


def main():
    p = argparse.ArgumentParser(description="Cadence WSL koprusu")
    p.add_argument("--distro", default=DEFAULT_DISTRO)
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="WSL erisimi + arac PATH kontrolu")
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

    args = p.parse_args()
    br = CadenceBridge(distro=args.distro, workdir=args.workdir)

    if args.cmd == "check":
        for key, val in br.check().items():
            print(f"{key:12s}: {val}")
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
