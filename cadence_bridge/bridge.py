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
# wsl -d Alma_EDA varsayilan olarak root acar; Cadence ayarlari ise
# tonxiao'nun ~/.bashrc'sindedir. Bu yuzden -u ile kullanici belirtilir.
DEFAULT_USER = "tonxiao"

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

    def __init__(self, distro=DEFAULT_DISTRO, workdir=DEFAULT_WORKDIR,
                 env_script=None, user=DEFAULT_USER):
        self.distro = distro
        self.workdir = workdir
        self.user = user
        # Cadence PATH/lisans ayarlarinizi iceren betik (orn. ~/.cadence_env.sh).
        # None ise: varsa ~/.cadence_env.sh otomatik kaynaklanir.
        self.env_script = env_script or os.environ.get("CADENCE_ENV_SCRIPT")
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
        env = (f"source {shlex.quote(self.env_script)}; " if self.env_script
               else "[ -f ~/.cadence_env.sh ] && source ~/.cadence_env.sh; ")
        shell_cmd = f"{env}cd {shlex.quote(self.workdir)} && {command}"

        # "-lic": login + interaktif kabuk. Interaktif bayragi onemli: cogu
        # ~/.bashrc dosyasi "interaktif degilsen cik" korumasiyla baslar ve
        # Cadence PATH ayarlari o korumanin arkasinda kalir.
        if self.inside_wsl:
            argv = ["bash", "-lic", shell_cmd]
        else:
            if not self.wsl_exe:
                raise RuntimeError(
                    "wsl.exe bulunamadı — bu komut Windows PowerShell'den ya da "
                    "WSL içinden çalıştırılmalı.")
            argv = [self.wsl_exe, "-d", self.distro, "-u", self.user,
                    "--", "bash", "-lic", shell_cmd]

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
            path = out.splitlines()[-1] if rc == 0 and out else ""
            if not path:
                report[tool] = "PATH'te yok"
            elif path.startswith(("/usr/sbin", "/usr/bin", "/bin", "/sbin")):
                # ornek: /usr/sbin/pvs Linux'un LVM komutudur, Cadence PVS degil
                report[tool] = f"{path}  (DIKKAT: sistem komutu, Cadence degil)"
            else:
                report[tool] = path
        return report

    def env_report(self):
        """Ortam teşhisi: PATH, rc dosyalarındaki Cadence izleri, kurulum dizini.

        'check' araçları bulamadığında bu rapor, PATH'in nereden gelmesi
        gerektiğini gösterir — kullanıcıya soru sormak yerine tek komutla
        toplanır.
        """
        probes = [
            ("which_virtuoso", "command -v virtuoso || echo YOK"),
            ("path", "echo $PATH | tr ':' '\\n' | grep -iE 'cadence|eda|IC6|SPECTRE|XCELIUM|GENUS' || echo 'PATH icinde cadence izi yok'"),
            ("bashrc", "grep -nE 'PATH|cadence|eda|CDS|LM_LICENSE|source' ~/.bashrc 2>/dev/null | head -30 || echo yok"),
            ("bash_profile", "grep -nE 'PATH|cadence|eda|CDS|LM_LICENSE|source' ~/.bash_profile 2>/dev/null | head -20 || echo yok"),
            ("kurulum", "ls /opt/eda/cadence 2>/dev/null || echo '/opt/eda/cadence yok'"),
            ("ic_bin", "ls -d /opt/eda/cadence/IC618/tools*/bin /opt/eda/cadence/IC618/tools/dfII/bin 2>/dev/null || echo 'IC618 bin bulunamadi'"),
            ("lisans", "env | grep -iE 'CDS|LM_LICENSE' || echo 'lisans degiskeni yok'"),
        ]
        report = {}
        for key, cmd in probes:
            try:
                _, out, err = self.run(cmd, timeout=60)
                report[key] = out or err or "(bos)"
            except Exception as exc:
                report[key] = f"HATA: {exc}"
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
