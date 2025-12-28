# -*- coding: utf-8 -*-
"""style.py

Jawna, testowalna logika stylów wykresów (bez „magii” w plotting.py).

Cel tej wersji:
- rozwiązać problem „znaczniki nakładają się w tych samych X” przez
  deterministyczne (lub losowe) rozstawianie markerów per-seria.

Matplotlib wspiera:
- markevery=int            -> marker co N punktów (zawsze te same indeksy),
- markevery=(start, step)  -> marker co step, ale zaczynając od innego start,
- markevery=[i1,i2,...]    -> markery tylko w wybranych indeksach.

W tym module implementujemy tryby:
- marker_mode="offset"  : markevery=(offset, stride), offset zależny od label
- marker_mode="random"  : markevery=list losowych indeksów (reproducowalnie)
- marker_mode="single"  : jeden marker na serię (index zależny od label)

Konfiguracja (w COMPARE_JOBS[*]["plot"]):
- base_lw, base_ms
- marker_mode: "offset" | "random" | "single" | "none"
- marker_stride: int (używane w trybie "offset")
- marker_target: int (docelowa liczba markerów; używane w "random" i jako fallback)
- marker_seed: int (seed dla "random"/"single" gdy chcesz inny rozkład)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import zlib
import numpy as np
import matplotlib as mpl


def _stable_hash32(label: str) -> int:
    """Stabilny hash (niezależny od PYTHONHASHSEED)."""
    return zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF


def _hash_to_index(label: str, n: int) -> int:
    """Deterministyczny hash -> indeks w [0, n)."""
    if n <= 0:
        return 0
    return int(_stable_hash32(label) % n)


def _build_default_palette() -> list[str]:
    # tab20 + tab20b + tab20c daje 60 „rozróżnialnych” kolorów (typowo wystarcza)
    cmaps = ["tab20", "tab20b", "tab20c"]
    out: list[str] = []
    for name in cmaps:
        cmap = mpl.colormaps.get(name)
        if cmap is None:
            continue
        for i in range(getattr(cmap, "N", 20)):
            out.append(mpl.colors.to_hex(cmap(i)))
    # fallback awaryjny
    if not out:
        out = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    return out


DEFAULT_COLORS = _build_default_palette()
DEFAULT_LINESTYLES = ["-", "--", "-.", ":"]
DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]


def normalize_style_dict(src: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalizuje i filtruje klucze stylu (dopasowane do plt.plot)."""
    if not isinstance(src, dict):
        return {}

    # aliasy kluczy -> canonical
    alias = {
        "linestyle": "ls",
        "linewidth": "lw",
        "markersize": "ms",
        "markerfacecolor": "mfc",
        "markeredgecolor": "mec",
        "markeredgewidth": "mew",
    }

    out: Dict[str, Any] = {}
    for k, v in src.items():
        kk = alias.get(k, k)
        out[kk] = v

    allowed = {
        "color",
        "ls", "lw", "lw_scale",
        "marker", "ms", "ms_scale",
        "alpha", "markevery", "zorder",
        "mfc", "mec", "mew",
    }
    return {k: v for k, v in out.items() if k in allowed}


def merge_styles(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if v is not None:
            out[k] = v
    return out


def default_style_for_label(label: str, npts: int) -> Dict[str, Any]:
    """Domyślny styl deterministyczny (kolor/linia/marker) po labelu."""
    color = DEFAULT_COLORS[_hash_to_index(label, len(DEFAULT_COLORS))]
    ls = DEFAULT_LINESTYLES[_hash_to_index(str(label) + "|ls", len(DEFAULT_LINESTYLES))]
    mk = DEFAULT_MARKERS[_hash_to_index(str(label) + "|mk", len(DEFAULT_MARKERS))]

    # bazowo: marker co ~60 punktów (ale bez offsetu -> offset dorabiamy w resolverze)
    if npts <= 0:
        me = 1
    else:
        me = max(1, int(npts // 60))

    return {"color": color, "ls": ls, "marker": mk, "markevery": me}


@dataclass(frozen=True)
class StyleResolver:
    """Rozwiązuje finalny styl dla etykiety."""
    style_map: Dict[str, Dict[str, Any]]
    base_lw: float = 1.2
    base_ms: float = 3.0

    # marker distribution
    marker_mode: str = "offset"      # offset|random|single|none
    marker_stride: Optional[int] = None
    marker_target: int = 12
    marker_seed: int = 0

    def _apply_marker_distribution(self, st: Dict[str, Any], label: str, npts: int) -> Dict[str, Any]:
        mk = st.get("marker", None)
        if mk is None or mk == "" or str(mk).lower() == "none":
            return st

        # jeśli user jawnie ustawił markevery w style_map -> respektujemy
        if isinstance(self.style_map.get(label, {}), dict) and ("markevery" in self.style_map[label]):
            return st

        mode = (self.marker_mode or "offset").lower()
        if mode in ("none", "off", "false", "0"):
            return st

        if npts <= 0:
            return st

        # stride (co ile punktów marker) – preferuj marker_stride, inaczej wylicz z target
        stride = self.marker_stride
        if stride is None:
            stride = max(1, int(round(npts / max(1, self.marker_target))))
        stride = max(1, int(stride))

        h = _stable_hash32(label)
        rng = np.random.default_rng(int(self.marker_seed) + int(h))

        if mode == "single":
            idx = int(rng.integers(0, npts)) if npts > 1 else 0
            st["markevery"] = [idx]
            return st

        if mode == "random":
            k = min(max(1, int(self.marker_target)), npts)
            if k >= npts:
                st["markevery"] = 1
                return st
            idx = np.sort(rng.choice(npts, size=k, replace=False)).tolist()
            st["markevery"] = idx
            return st

        # default: "offset"
        if stride == 1:
            st["markevery"] = 1
            return st
        offset = int(h % stride)
        st["markevery"] = (offset, stride)
        return st

    def style_for(self, label: str, npts: int) -> Dict[str, Any]:
        base = default_style_for_label(label, npts)
        base.update({"lw": float(self.base_lw), "ms": float(self.base_ms)})

        override = normalize_style_dict(self.style_map.get(label))
        st = merge_styles(base, override)

        # scales (applied after override)
        if "lw_scale" in st:
            try:
                st["lw"] = float(st.get("lw", self.base_lw)) * float(st["lw_scale"])
            except Exception:
                pass
        if "ms_scale" in st:
            try:
                st["ms"] = float(st.get("ms", self.base_ms)) * float(st["ms_scale"])
            except Exception:
                pass

        # marker distribution (offset/random/single)
        st = self._apply_marker_distribution(st, label, npts)
        return st

    @staticmethod
    def from_job(job: Optional[Dict[str, Any]]) -> "StyleResolver":
        j = job or {}
        style_map = j.get("style_map", {}) or {}
        plot_cfg = j.get("plot", {}) or {}

        base_lw = float(plot_cfg.get("base_lw", 1.2))
        base_ms = float(plot_cfg.get("base_ms", 3.0))

        marker_mode = str(plot_cfg.get("marker_mode", "offset"))
        marker_stride = plot_cfg.get("marker_stride", None)
        marker_stride = int(marker_stride) if marker_stride is not None else None
        marker_target = int(plot_cfg.get("marker_target", 12))
        marker_seed = int(plot_cfg.get("marker_seed", 0))

        # normalize all entries once
        norm_map: Dict[str, Dict[str, Any]] = {}
        for k, v in style_map.items():
            norm_map[str(k)] = normalize_style_dict(v)

        return StyleResolver(
            style_map=norm_map,
            base_lw=base_lw,
            base_ms=base_ms,
            marker_mode=marker_mode,
            marker_stride=marker_stride,
            marker_target=marker_target,
            marker_seed=marker_seed,
        )
