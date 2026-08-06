# -*- coding: utf-8 -*-
"""Cadence araç köprüsü — Windows'tan WSL (Alma_EDA) içindeki EDA araçlarına.

Bu modül, ileride bir YZ ajanının "elleri" olacak katmandır: model hangi
komutu isterse istesin, araçlara DOKUNAN tek yer burasıdır. Şimdilik ajan
yok; komutlar insan tarafından (demo.py CLI'ı ile) tetiklenir.

Windows PowerShell'den çağrıldığında komutlar şu kalıpla WSL'e geçer:

    wsl -d Alma_EDA -- bash -lc "cd <workdir> && <komut>"

``bash -lc`` login kabuğu açar; böylece ~/.bashrc'nizdeki Cadence PATH /
lisans ayarları yüklenir. WSL içinden (Alma_EDA'nın kendisinden)
çalıştırılırsa ``wsl`` sarmalayıcısı atlanır, komut doğrudan koşar.
"""

import os
import shlex
import shutil
import subprocess

DEFAULT_DISTRO = "Alma_EDA"
DEFAULT_WORKDIR = "/home/tonxiao/work65"

# Araç adı -> sürüm bayrağı (kurulum kontrolünde kullanılır)
KNOWN_TOOLS = {
    "virtuoso": "-W",       # IC618 (analog/şematik/layout)
    "spectre": "-W",        # SPECTRE241 (devre simülatörü)
    "ocean": None,          # OCEAN batch (sürüm bayrağı yok, which yeter)
    "genus": "-version",    # sentez (dijital)
    "innovus": "-version",  # yerleşim/serim (dijital)
    "xrun": "-version",     # XCELIUM simülasyon (dijital)
    "modus": "-version",    # test (DFT)
    "quantus": "-version",  # parazit çıkarımı
    "pvs": "-version",      # fiziksel doğrulama
}


class CadenceBridge:
    """WSL içindeki Cadence araçlarına komut gönderen köprü."""

    def __init__(self, distro=DEFAULT_DISTRO, workdir=DEFAULT_WORKDIR):
        self.distro = distro
        self.workdir = workdir
        # WSL içinde miyiz, Windows'ta mı? (WSL'de /proc/version 'microsoft' içerir)
        self.inside_wsl = False
        try:
            with open("/proc/version") as f:
                self.inside_wsl = "microsoft" in f.read().lower()
        except OSError:
            pass
        self.wsl_exe = shutil.which("wsl") or shutil.which("wsl.exe")

    # ------------------------------------------------------------------ temel
    def run(self, command, timeout=600, background=False):
        """Komutu workdir içinde çalıştırır; (returncode, stdout, stderr) döner.

        background=True: komut nohup ile koparılır (GUI araçları için) —
        dönüş hemen gelir, araç açık kalır.
        """
        if background:
            command = f"nohup {command} >/dev/null 2>&1 & disown; echo BASLATILDI"
        shell_cmd = f"cd {shlex.quote(self.workdir)} && {command}"

        if self.inside_wsl:
            argv = ["bash", "-lc", shell_cmd]
        else:
            if not self.wsl_exe:
                raise RuntimeError(
                    "wsl.exe bulunamadı — bu komut Windows PowerShell'den ya da "
                    "WSL içinden çalıştırılmalı.")
            argv = [self.wsl_exe, "-d", self.distro, "--", "bash", "-lc", shell_cmd]

        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8",
                              errors="replace")
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

    # ------------------------------------------------------------- kontroller
    def check(self):
        """WSL erişimini ve bilinen araçların PATH'te olup olmadığını raporlar."""
        report = {}
        try:
            rc, out, err = self.run("echo OK && pwd", timeout=60)
            report["wsl"] = f"erisim OK, workdir: {out.splitlines()[-1]}" if rc == 0 \
                else f"HATA: {err or out}"
        except Exception as exc:
            report["wsl"] = f"HATA: {exc}"
            return report
        for tool in KNOWN_TOOLS:
            rc, out, _ = self.run(f"command -v {tool}", timeout=60)
            report[tool] = out if rc == 0 and out else "PATH'te yok"
        return report

    # ---------------------------------------------------------------- analog
    def launch_virtuoso(self):
        """Virtuoso GUI'yi workdir içinde başlatır (virtuoso -64 &)."""
        return self.run("virtuoso -64", background=True)

    def run_skill(self, script_path, log="skill_run.log"):
        """SKILL betiğini başsız (GUI'siz) Virtuoso'da çalıştırır."""
        return self.run(
            f"virtuoso -nograph -replay {shlex.quote(script_path)} "
            f"-log {shlex.quote(log)}")

    def run_ocean(self, script_path):
        """OCEAN betiğini batch modda çalıştırır (simülasyon otomasyonu)."""
        return self.run(f"ocean -nograph -replay {shlex.quote(script_path)}")

    def run_spectre(self, netlist_path):
        """Spectre'ı doğrudan bir netlist üzerinde koşar."""
        return self.run(f"spectre {shlex.quote(netlist_path)}")

    # ---------------------------------------------------------------- dijital
    def run_tcl(self, tool, script_path):
        """Dijital araçları (genus/innovus/modus...) TCL betiğiyle batch koşar."""
        return self.run(
            f"{shlex.quote(tool)} -no_gui -batch -files {shlex.quote(script_path)}")

    def run_xrun(self, args):
        """XCELIUM (xrun) simülasyonu — args serbest metin (ör. 'tb.v -access +rwc')."""
        return self.run(f"xrun {args}")
