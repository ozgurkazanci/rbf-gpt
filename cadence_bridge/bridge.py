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

import base64
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
    "virtuoso": "-W",       # IC231 (analog/şematik/layout)
    "spectre": "-W",        # SPECTRE241 (devre simülatörü)
    "ocean": None,          # OCEAN batch (sürüm bayrağı yok, which yeter)
    "genus": "-version",    # sentez (dijital)
    "innovus": "-version",  # yerleşim/serim (dijital)
    "xrun": "-version",     # XCELIUM simülasyon (dijital)
    "modus": "-version",    # test (DFT)
    "quantus": "-version",  # parazit çıkarımı
    "pvs": "-version",      # fiziksel doğrulama
    "tempus": "-version",   # SSV231: statik zamanlama signoff
    "voltus": "-version",   # SSV231: güç bütünlüğü signoff
    "lec": None,            # CONFRML232: Conformal eşdeğerlik kontrolü
}

# Süreç tasarım kitleri (PDK) kök dizini — 65nm kitler burada yaşar.
PDK_ROOT = "/opt/eda/PDK"

# ~/.bashrc'de PATH'e eklenmemiş kurulumlar: köprü, bu dizinlerden var
# olanları her komuttan önce PATH'e ekler (SSV: tempus/voltus, Conformal).
EXTRA_PATH_DIRS = [
    "/opt/eda/cadence/SSV231/bin",
    "/opt/eda/cadence/SSV231/tools.lnx86/bin",
    "/opt/eda/cadence/SSV231/tools/bin",
    "/opt/eda/cadence/CONFRML232/bin",
    "/opt/eda/cadence/CONFRML232/tools.lnx86/bin",
    "/opt/eda/cadence/CONFRML232/tools/bin",
]


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
        # ~/.bashrc her zaman elle kaynaklanir: stdin'den beslenen login kabugu
        # .bash_profile -> .bashrc zincirine guvenmek zorunda kalmasin.
        env = "[ -f ~/.bashrc ] && source ~/.bashrc; "
        env += (f"source {shlex.quote(self.env_script)}; " if self.env_script
                else "[ -f ~/.cadence_env.sh ] && source ~/.cadence_env.sh; ")
        extra = " ".join(shlex.quote(d) for d in EXTRA_PATH_DIRS)
        env += (f'for _d in {extra}; do [ -d "$_d" ] && PATH="$PATH:$_d"; done; '
                "export PATH; ")
        shell_cmd = f"{env}cd {shlex.quote(self.workdir)} && {command}"

        # Betik, argüman yerine STDIN üzerinden bash'e akıtılır. Windows ->
        # wsl.exe -> bash argüman geçişi tırnak/özel karakterleri bozuyor
        # (CommandLineToArgvW + wsl'nin yeniden birleştirmesi); stdin yolunda
        # aktarılan argümanlar sadece sabit bayraklar olduğundan bozulacak
        # hiçbir şey kalmaz. "-l" login kabuğudur; .bashrc yukarıda elle
        # kaynaklandığı için ortam her durumda yüklenir.
        if self.inside_wsl:
            argv = ["bash", "-l", "-s"]
        else:
            if not self.wsl_exe:
                raise RuntimeError(
                    "wsl.exe bulunamadı — bu komut Windows PowerShell'den ya da "
                    "WSL içinden çalıştırılmalı.")
            argv = [self.wsl_exe, "-d", self.distro, "-u", self.user,
                    "--", "bash", "-l", "-s"]

        proc = subprocess.run(argv, input=shell_cmd, capture_output=True,
                              text=True, timeout=timeout, encoding="utf-8",
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

    def list_library(self, lib):
        """Cadence kütüphanesindeki hücreleri ve görünümlerini listeler.

        lib: kütüphane adı (workdir altında aranır) veya mutlak yol.
        """
        path = lib if lib.startswith("/") else f"{self.workdir}/{lib}"
        q = shlex.quote(path)
        cmd = (
            f'if [ ! -d {q} ]; then echo "KUTUPHANE YOK: {path}"; exit 1; fi; '
            f'for c in {q}/*/; do b=$(basename "$c"); '
            f'case "$b" in .*) continue;; esac; '
            f'echo "$b : $(ls "$c" 2>/dev/null | tr "\\n" " ")"; done'
        )
        return self.run(cmd, timeout=120)

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

    def pdk_report(self, root=PDK_ROOT):
        """PDK envanteri: kitler, Spectre model dosyaları, köşe bölümleri.

        Gerçek transistörlü simülasyon yazabilmek için gereken üç bilgiyi
        toplar: hangi kitler var, spectre model .scs dosyaları nerede ve
        model dosyalarında hangi section (tt/ss/ff) adları geçiyor.
        """
        q = shlex.quote(root)
        probes = [
            ("kitler", f"ls {q} 2>/dev/null || echo 'PDK koku yok: {root}'"),
            ("model_dizinleri",
             f"find {q} -maxdepth 5 -type d \\( -iname '*model*' -o -iname 'spectre' \\) 2>/dev/null | head -20"),
            ("scs_dosyalari",
             f"find {q} -maxdepth 7 -iname '*.scs' 2>/dev/null | head -40"),
            ("kose_ornekleri",
             f"for f in $(find {q} -maxdepth 7 -iname '*.scs' 2>/dev/null | head -5); do "
             f"echo \"--- $f\"; grep -m 6 -E '^section|^simulator|include' \"$f\" 2>/dev/null; done"),
            # Dijital taraf (PDK/65): stdcell Liberty/LEF/Verilog koleksiyonu
            ("liberty_lib",
             f"find {q} -maxdepth 8 \\( -iname '*.lib' -o -iname '*.lib.gz' \\) 2>/dev/null | head -25"),
            ("lef",
             f"find {q} -maxdepth 8 \\( -iname '*.lef' -o -iname '*.tlef' \\) 2>/dev/null | head -15"),
            ("verilog_model",
             f"find {q} -maxdepth 8 -iname '*.v' 2>/dev/null | head -15"),
        ]
        report = {}
        for key, cmd in probes:
            try:
                _, out, err = self.run(cmd, timeout=180)
                report[key] = out or err or "(bos)"
            except Exception as exc:
                report[key] = f"HATA: {exc}"
        return report

    # Köprü doğrulama devresi: RC alçak geçiren filtre (PDK gerektirmez).
    # tau = R*C = 1us; kaynak 1us'de 0->1V basar, out 10us'de ~1V'a oturur.
    FIRST_SIM_NETLIST = """\
// cadence_bridge dogrulama devresi: RC filtre, tau=1us
simulator lang=spectre
v1 (in 0) vsource type=pulse val0=0 val1=1 delay=1u rise=1n
r1 (in out) resistor r=1k
c1 (out 0) capacitor c=1n
tran1 tran stop=10u
saveOptions options save=allpub
"""

    def first_sim(self):
        """Spectre'la uçtan uca ilk simülasyon: netlist yaz, koş, sonucu dök.

        Çıktının sonunda psfascii formatındaki tran verisi bulunur; demo.py
        bunu ayrıştırıp V(out) son değerini fizikle karşılaştırır.
        """
        d = f"{self.workdir}/bridge_test"
        script = (
            f"mkdir -p {d} && cd {d} && "
            f"cat > rc_test.scs <<'NETLIST_EOF'\n{self.FIRST_SIM_NETLIST}NETLIST_EOF\n"
            f"spectre rc_test.scs -format psfascii -raw rc_test.raw "
            f"> spectre_run.log 2>&1; echo SPECTRE_RC=$?; "
            f"tail -3 spectre_run.log; echo ---RAW---; "
            f"tail -60 rc_test.raw/tran1.tran 2>/dev/null || echo RAW_YOK")
        return self.run(script, timeout=600)

    # ---------------------------------- gerçek PDK ile evirici simülasyonu
    MODELS_ONLINE = f"{PDK_ROOT}/CRN65GPNEW/CRN65GPNEW/models/online"

    def discover_models(self):
        """PDK'daki model klasörünü, tt köşesini ve köşe dosyasını bulur.

        Sonuç önbelleğe alınır: ajan döngüsü yüzlerce simülasyon koşarken
        keşif her seferinde tekrarlanmaz.
        """
        if getattr(self, "_models_cache", None):
            return self._models_cache
        probe = (
            f"ls {self.MODELS_ONLINE} 2>/dev/null; echo ===; "
            f"for d in {self.MODELS_ONLINE}/*/spectre; do echo DIR:$d; "
            f"grep -h '^section' $d/cor_std_mos.scs $d/cor.scs 2>/dev/null "
            f"| head -12; done")
        rc, out, err = self.run(probe, timeout=120)
        if rc != 0 or "DIR:" not in out:
            return {"hata": err or out or "model klasoru bulunamadi"}
        lines = out.split("DIR:")[1].splitlines()
        sdir = lines[0].strip()
        sections = [ln.split()[1] for ln in lines[1:]
                    if ln.startswith("section") and len(ln.split()) > 1]
        sec = next((s for s in sections if s.startswith("tt")), None)
        rc, cor, _ = self.run(
            f"ls {sdir}/cor_std_mos.scs 2>/dev/null || ls {sdir}/cor.scs")
        cor = cor.strip().splitlines()[-1] if cor else f"{sdir}/cor.scs"
        info = {"model_dir": sdir, "section": sec, "corner_file": cor}
        if sec:
            self._models_cache = info
        return info

    def inverter_vtc(self, wn="200n", wp="400n", length="60n", vdd=1.2,
                     nmos="nch", pmos="pch", corner=None, section=None,
                     points=121, tag="agent"):
        """Verilen boyutlarla evirici DC taraması koşar; ham çıktı döner.

        Ajanın parametre araması bu metodu tekrar tekrar çağırır; her çağrı
        kendi dosya adını (tag) kullanır ki paralel/ardışık koşular
        birbirinin sonucunu ezmesin.
        """
        if corner is None or section is None:
            info = self.discover_models()
            if info.get("hata") or not info.get("section"):
                return 1, "", info.get("hata", "tt kosesi yok")
            corner = corner or info["corner_file"]
            section = section or info["section"]

        netlist = f"""// TSMC65 CMOS evirici — ajan parametre taramasi
simulator lang=spectre
include "{corner}" section = {section}
vdd (vdd 0) vsource dc={vdd}
vin (in 0) vsource dc=0
mp (out in vdd vdd) {pmos} w={wp} l={length}
mn (out in 0 0) {nmos} w={wn} l={length}
vtc dc dev=vin param=dc start=0 stop={vdd} lin={points}
saveOptions options save=allpub
"""
        d = f"{self.workdir}/bridge_test"
        f = f"inv_{tag}"
        script = (
            f"mkdir -p {d} && cd {d} && rm -rf {f}.raw && "
            f"cat > {f}.scs <<'NETLIST_EOF'\n{netlist}NETLIST_EOF\n"
            f"spectre {f}.scs -format psfascii -raw {f}.raw "
            f"> {f}.log 2>&1; echo SPECTRE_RC=$?; "
            f"grep -m 4 -iE 'error|fatal' {f}.log; echo ---RAW---; "
            + f"grep -E '^\"(in|out)\"' {f}.raw/vtc.dc 2>/dev/null "
            + "|| echo RAW_YOK")
        return self.run(script, timeout=900)

    def inverter_sim(self):
        """TSMC65 modelleriyle CMOS evirici DC taraması (VTC) — 4 adım.

        1) models/online altındaki voltaj klasörlerini ve köşe dosyalarını
           keşfet; 2) tt köşesini seç; 3) netlist'i üretip Spectre'da koş;
        4) psfascii çıktısını dök (Vm hesabı demo.py'de yapılır).
        Döner: (info_dict, rc, out, err) — out, ham VTC verisini içerir.
        """
        info = {}
        # --- 1) keşif: voltaj klasörleri + köşe bölümleri -------------------
        probe = (
            f"ls {self.MODELS_ONLINE} 2>/dev/null; echo ===; "
            f"for d in {self.MODELS_ONLINE}/*/spectre; do echo DIR:$d; "
            f"grep -h '^section' $d/cor_std_mos.scs $d/cor.scs 2>/dev/null "
            f"| head -12; done")
        rc, out, err = self.run(probe, timeout=120)
        if rc != 0 or "DIR:" not in out:
            return info, rc, out, (err or "model klasoru bulunamadi")

        # İlk klasörü kullan (kitte tek model seti var; ad "2.5V" olsa bile
        # cor_std_mos.scs'in düz tt bölümü ÇEKİRDEK 1.2V cihazları kapsar).
        blocks = out.split("DIR:")[1:]
        lines = blocks[0].splitlines()
        sdir = lines[0].strip()
        sections = [ln.split()[1] for ln in lines[1:]
                    if ln.startswith("section") and len(ln.split()) > 1]
        # tt ile başlayan bölümü seç (tt, tt_std_mos, ...)
        sec = next((s for s in sections if s.startswith("tt")), None)
        info.update(model_dir=sdir, section=sec)
        if not sec:
            return info, 1, out, "tt kosesi bulunamadi — cikti ile bakalim"

        # köşe dosyası: cor_std_mos.scs varsa onu, yoksa cor.scs
        rc, cor, _ = self.run(
            f"ls {sdir}/cor_std_mos.scs 2>/dev/null || ls {sdir}/cor.scs")
        cor = cor.strip().splitlines()[-1] if cor else f"{sdir}/cor.scs"
        info["corner_file"] = cor

        # --- 3) netlist üret + koş: önce core, olmazsa IO cihazları --------
        attempts = [
            {"nmos": "nch", "pmos": "pch", "l": "60n",
             "wn": "200n", "wp": "400n", "vdd": 1.2},
            {"nmos": "nch_25", "pmos": "pch_25", "l": "280n",
             "wn": "1u", "wp": "2u", "vdd": 2.5},
        ]
        d = f"{self.workdir}/bridge_test"
        last = (1, "", "")
        for att in attempts:
            netlist = f"""// TSMC65 CMOS evirici — VTC taramasi (kopru 2. tur)
simulator lang=spectre
include "{cor}" section = {sec}
vdd (vdd 0) vsource dc={att['vdd']}
vin (in 0) vsource dc=0
mp (out in vdd vdd) {att['pmos']} w={att['wp']} l={att['l']}
mn (out in 0 0) {att['nmos']} w={att['wn']} l={att['l']}
vtc dc dev=vin param=dc start=0 stop={att['vdd']} lin=121
saveOptions options save=allpub
"""
            script = (
                f"mkdir -p {d} && cd {d} && rm -rf inv_vtc.raw && "
                f"cat > inv_vtc.scs <<'NETLIST_EOF'\n{netlist}NETLIST_EOF\n"
                f"spectre inv_vtc.scs -format psfascii -raw inv_vtc.raw "
                f"> inv_run.log 2>&1; echo SPECTRE_RC=$?; "
                f"grep -m 6 -iE 'error|fatal' inv_run.log; "
                f"echo ---RAW---; "
                # Sadece in/out satirlari suzulur: kirpma yok, tum sweep
                # noktalari eksiksiz gelir (tail ile kirpmak sweep'in basini
                # yiyip yanlis VTC'ye yol aciyordu).
                + 'grep -E \'^"(in|out)"\' inv_vtc.raw/vtc.dc 2>/dev/null '
                + "|| echo RAW_YOK")
            rc, out, err = self.run(script, timeout=900)
            info.update(nmos=att["nmos"], pmos=att["pmos"], vdd=att["vdd"])
            last = (rc, out, err)
            if "SPECTRE_RC=0" in out and '"out"' in out:
                return info, rc, out, err
            info[f"deneme_{att['nmos']}"] = "basarisiz (log satirlari ciktida)"
        return info, last[0], last[1], last[2]

    # ---------------------------------------------------------------- dijital
    # RTL doğrulama tasarımı: 8-bit sayıcı + kendini-denetleyen testbench.
    # PDK/stdcell gerektirmez — saf RTL simülasyonu (xrun) yeterlidir.
    RTL_DESIGN = """\
// sayici.v — 8-bit senkron sayici (kopru RTL dogrulama tasarimi)
`timescale 1ns/1ps
module sayici (
  input  wire       clk,
  input  wire       rst_n,
  input  wire       en,
  output reg  [7:0] q
);
  always @(posedge clk or negedge rst_n)
    if (!rst_n)      q <= 8'd0;
    else if (en)     q <= q + 8'd1;
endmodule
"""

    RTL_TB = """\
// sayici_tb.v — kendini denetleyen testbench: reset, sayma, durdurma, tasma
`timescale 1ns/1ps
module sayici_tb;
  reg clk = 0, rst_n = 0, en = 0;
  wire [7:0] q;
  integer hata = 0;

  sayici dut(.clk(clk), .rst_n(rst_n), .en(en), .q(q));
  always #5 clk = ~clk;

  task kontrol(input [7:0] bekle, input [127:0] ad);
    if (q !== bekle) begin
      $display("HATA [%0s]: q=%0d beklenen=%0d (t=%0t)", ad, q, bekle, $time);
      hata = hata + 1;
    end else
      $display("OK   [%0s]: q=%0d (t=%0t)", ad, q, $time);
  endtask

  initial begin
    #12 rst_n = 1;               // reset birak
    kontrol(8'd0, "reset");
    en = 1;                      // 10 cevrim say
    repeat (10) @(posedge clk);
    #1 kontrol(8'd10, "sayma");
    en = 0;                      // dur: deger korunmali
    repeat (5) @(posedge clk);
    #1 kontrol(8'd10, "durdurma");
    en = 1;                      // 246 cevrim daha: 256 -> tasma -> 0
    repeat (246) @(posedge clk);
    #1 kontrol(8'd0, "tasma");
    if (hata == 0) $display("TB_SONUC: PASS (4/4 kontrol gecti)");
    else           $display("TB_SONUC: FAIL (%0d hata)", hata);
    $finish;
  end
endmodule
"""

    def rtl_sim(self):
        """Xcelium (xrun) ile uçtan uca RTL simülasyonu — dijital ilk tur.

        Sayıcı tasarımını ve testbench'i workdir'e yazar, xrun ile derleyip
        simüle eder; testbench'in TB_SONUC satırı sonucu taşır.
        """
        d = f"{self.workdir}/bridge_test/rtl"
        script = (
            f"command -v xrun >/dev/null || {{ echo XRUN_YOK; exit 1; }}; "
            f"mkdir -p {d} && cd {d} && "
            f"cat > sayici.v <<'RTL_EOF'\n{self.RTL_DESIGN}RTL_EOF\n"
            f"cat > sayici_tb.v <<'RTL_EOF'\n{self.RTL_TB}RTL_EOF\n"
            f"xrun -q sayici.v sayici_tb.v 2>&1 | tail -30")
        return self.run(script, timeout=900)

    # ------------------------------------------------------------ tcl/batch
    def run_tcl(self, tool, script_path):
        """Dijital araçları (genus/innovus/modus...) TCL betiğiyle batch koşar."""
        return self.run(
            f"{shlex.quote(tool)} -no_gui -batch -files {shlex.quote(script_path)}")

    def run_xrun(self, args):
        """XCELIUM (xrun) simülasyonu — args serbest metin (ör. 'tb.v -access +rwc')."""
        return self.run(f"xrun {args}")
