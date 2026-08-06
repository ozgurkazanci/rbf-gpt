# -*- coding: utf-8 -*-
"""psfascii çıktısını okuma yardımcıları.

Spectre'ın psfascii biçimi her sweep noktası için sinyalleri
``"ad" deger`` satırları hâlinde yazar. Buradaki fonksiyonlar hem CLI
(demo.py) hem de ajan (agent.py) tarafından kullanılır.
"""

import re


def parse_signals(raw):
    """Ham metinden {sinyal_adi: [degerler]} çıkarır."""
    sig = {}
    for name, val in re.findall(r'"([^"]+)"\s+([-+0-9.eE]+)\s*$', raw,
                                re.MULTILINE):
        try:
            sig.setdefault(name, []).append(float(val))
        except ValueError:
            continue
    return sig


def vtc_metrics(raw):
    """Evirici DC taramasından ölçümler çıkarır.

    Döner: {"n", "vin", "vout", "voh", "vol", "vm", "gain"} veya hata
    durumunda {"hata": "..."}.
    """
    sig = parse_signals(raw)
    vin, vout = sig.get("in", []), sig.get("out", [])
    n = min(len(vin), len(vout))
    if n < 10:
        return {"hata": f"yetersiz veri (in={len(vin)}, out={len(vout)})"}
    vin, vout = vin[:n], vout[:n]

    # Anahtarlama eşiği: vout(vin) eğrisinin vout = vin doğrusunu kestiği yer
    vm = None
    for i in range(1, n):
        d0, d1 = vout[i - 1] - vin[i - 1], vout[i] - vin[i]
        if d0 * d1 <= 0:
            t = d0 / (d0 - d1) if d0 != d1 else 0.5
            vm = vin[i - 1] + t * (vin[i] - vin[i - 1])
            break

    # Geçiş bölgesindeki en dik eğim (kazanç)
    gain = 0.0
    for i in range(1, n):
        dv = vin[i] - vin[i - 1]
        if dv:
            gain = min(gain, (vout[i] - vout[i - 1]) / dv)

    return {"n": n, "vin": vin, "vout": vout, "voh": vout[0],
            "vol": vout[-1], "vm": vm, "gain": gain}
