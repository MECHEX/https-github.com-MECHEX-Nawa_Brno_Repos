# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import math

from constants import AXIS_CANDIDATES

STEP_RE = re.compile(r"_S(\d{4,})", re.IGNORECASE)

# ---------- małe utilsy ----------
def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def pick_axis_col(df) -> Optional[str]:
    for c in AXIS_CANDIDATES:
        if c in df.columns:
            return c
    return None

def extract_step(stem: str) -> int:
    m = STEP_RE.search(stem)
    return int(m.group(1)) if m else -1

def fluid_from_name(stem: str) -> str:
    s = stem.lower()
    if "fluid1" in s or "_f1_" in s or s.endswith("_f1") or "f1_" in s:
        return "Fluid1"
    if "fluid2" in s or "_f2_" in s or s.endswith("_f2") or "f2_" in s:
        return "Fluid2"
    return "Fluid2" if "_f2" in s else "Fluid1"

def normalize_prefixes(text: str) -> str:
    t = re.sub(r"\bf[12]_wall_band_", "wall_band_", text)
    t = re.sub(r"\bf[12]_(?:ziso_|yiso_)", "ziso_", t)
    t = re.sub(r"\bf[12]_env_", "env_", t)
    return t

def fmt_scalar_short(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.3f}".rstrip("0").rstrip(".")

def fmt_time_tag(t0: float, t1: float) -> str:
    return f"t{fmt_scalar_short(t0)}-{fmt_scalar_short(t1)}s"

def fmt_dt_tag(dt: float) -> str:
    if dt == 0.0:
        return "0s"
    exp = int(math.floor(math.log10(abs(dt))))
    mant = dt / (10 ** exp)
    return f"1e{exp}s" if abs(mant - 1.0) < 1e-12 else f"{dt:.0e}s"

def parts_tag(parts: List[str]) -> str:
    nums = []
    for p in parts:
        m = re.search(r"part\s*([0-9]+)", p, re.IGNORECASE)
        nums.append(int(m.group(1)) if m else -1)
    nums = sorted([n for n in nums if n >= 0])
    return ("p" + "".join(map(str, nums))) if nums else "p"

def outfile_suffix(fluid: str, parts: List[str], t0: float, t1: float, dt: float) -> str:
    """Zwraca sufiks do nazw plików: 'F2_p3_t10-11s_1e-5s'."""
    ftag = "F1" if fluid == "Fluid1" else "F2"
    ptag = parts_tag(parts)
    ttag = fmt_time_tag(t0, t1)
    dtag = fmt_dt_tag(dt)
    return f"{ftag}_{ptag}_{ttag}_{dtag}"

def mode_tag(n_steps: Optional[int], duration_s: Optional[float]) -> str:
    if n_steps is not None:
        return f"N{int(n_steps)}"
    if duration_s is not None:
        return f"D{fmt_scalar_short(float(duration_s))}s"
    return "ALL"

# ---------- SRP indeks ----------
def collect_all_data(base_dir: Path, PARTS: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, List[Tuple[float, str, Path]]]]:
    """
    parts -> fluid -> [(t_local[s], part_name, path)] posortowane po czasie lokalnym.
    t_local = step * dt_sim_s (dla danego partu).
    """
    out: Dict[str, Dict[str, List[Tuple[float, str, Path]]]] = {}
    for part, meta in PARTS.items():
        dt = float(meta["dt_sim_s"])
        pdir = base_dir / part
        if not pdir.exists():
            continue
        for p in pdir.glob("*.srp"):
            step = extract_step(p.stem)
            if step < 0:
                continue
            t_local = step * dt
            fluid = fluid_from_name(p.stem)
            out.setdefault(part, {}).setdefault(fluid, []).append((t_local, part, p))
    for part in out:
        for fluid in out[part]:
            out[part][fluid].sort(key=lambda t: t[0])
    return out

def filter_by_local_window(items: List[Tuple[float, str, Path]], t_len: float) -> List[Tuple[float, str, Path]]:
    return [t for t in items if (-1e-12) <= t[0] <= (t_len + 1e-12)]

def filter_by_global_window(items: List[Tuple[float, str, Path]], t_start: float, t0: float, t1: float) -> List[Tuple[float, str, Path]]:
    """Filtr wg CZASU GLOBALNEGO: t_global = t_start + t_local ∈ [t0, t1]."""
    out = []
    for (t_local, part_name, path) in items:
        t_global = t_start + t_local
        if (t0 - 1e-12) <= t_global <= (t1 + 1e-12):
            out.append((t_local, part_name, path))
    return out

# ---------- okna czasu dla jobów ----------
def resolve_time_window_for_job(parts: List[str], PARTS: Dict[str, Dict[str, float]],
                                t0_s: Optional[float], n_steps: Optional[int], duration_s: Optional[float]) -> Tuple[str, float, float, float]:
    """
    Zwraca (mode, t0, t1, dt_ref).
    - mode: 'ALL' / 'N...' / 'D...'
    - t0,t1: do użycia w nazwach i filtrze globalnym (dla N/D)
    - dt_ref: referencyjne dt do etykiet nazw
    Walidacja: dokładnie jeden z (n_steps, duration_s) lub żaden.
    """
    # referencyjne dt z pierwszego partu (ostrzeganie o niespójności zostaw w main)
    dt_ref = float(PARTS[parts[0]]["dt_sim_s"]) if parts else 0.0

    if (n_steps is None) and (duration_s is None):
        # ALL → użyj pełnego zakresu (t0,t1 tylko do nazwy folderu/raportu)
        tmins = [float(PARTS[p]["t_start_s"]) for p in parts if p in PARTS]
        tmaxs = [float(PARTS[p]["t_end_s"])   for p in parts if p in PARTS]
        t0 = min(tmins) if tmins else 0.0
        t1 = max(tmaxs) if tmaxs else 0.0
        return ("ALL", t0, t1, dt_ref)

    if (n_steps is not None) and (duration_s is not None):
        raise ValueError("Podaj dokładnie jeden z: n_steps, duration_s (lub żaden dla ALL).")

    if t0_s is None:
        raise ValueError("Dla trybów N/D wymagany jest t0_s.")

    if n_steps is not None:
        if int(n_steps) <= 0:
            raise ValueError("n_steps musi być > 0.")
        t0 = float(t0_s); t1 = t0 + int(n_steps) * dt_ref
        return (mode_tag(n_steps, None), t0, t1, dt_ref)

    # duration_s
    if float(duration_s) <= 0.0:
        raise ValueError("duration_s musi być > 0.")
    t0 = float(t0_s); t1 = t0 + float(duration_s)
    return (mode_tag(None, duration_s), t0, t1, dt_ref)
