# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import re
from srp_types import SRPData

# --- regexy i pomocnicze ---
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
RE_DASH = re.compile(r"-{3,}")  # linia z myślnikami
RE_ROW  = re.compile(r"\s*([\w]+(?:_\d+)?)\s+(" + _NUM + r")\s*$")

def _collect_rows(lines: list[str], dash_idx: int) -> Tuple[Dict[str, float], int]:
    """
    Oczekuje indeksu linii z '----'. Zwraca (słownik, next_index_po_tabeli).
    """
    out: Dict[str, float] = {}
    i = dash_idx + 1
    n = len(lines)
    while i < n:
        # koniec tabeli: kolejna linia z '---' lub pusty wiersz nagłówka 'Net' itp.
        if RE_DASH.search(lines[i]):
            i += 1
            if i < n and lines[i].strip().lower().startswith("net"):
                i += 1
            break
        m = RE_ROW.match(lines[i])
        if m:
            key = m.group(1)
            try:
                val = float(m.group(2))
                out[key] = val
            except Exception:
                pass
        i += 1
    return out, i

def _norm_key(k: str) -> str:
    """
    Normalizuje identyfikatory z SRP do wspólnej postaci:
      - (f1_|f2_|F2_|...)?(x|y|z)iso_<n>  -> ziso_<n>
      - (f1_|f2_|...)?wall_band_<n>      -> wall_band_<n>
    Dodatkowo zamienia myślniki na podkreślenia i robi casefold.
    """
    s = k.strip().replace("-", "_")
    low = s.lower()

    m = re.match(r'^(?:f\d_)?([xyz])iso_(\d+)$', low)
    if m:
        # Oś nas nie interesuje w dalszych obliczeniach – liczy się indeks <n>
        return f"ziso_{m.group(2)}"

    m = re.match(r'^(?:f\d_)?wall_band_(\d+)$', low)
    if m:
        return f"wall_band_{m.group(1)}"

    return s  # bez zmian, gdy nic nie pasuje

def _norm_table(d: dict) -> dict:
    return {_norm_key(k): v for k, v in d.items()}

def _norm_table(d: Dict[str, float]) -> Dict[str, float]:
    return {_norm_key(k): v for k, v in d.items()}

# --- parser główny ---
def parse_srp(path: Path, *, verbose: bool = False) -> SRPData:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()
    n = len(lines)

    # płaszczyzny (ziso_i)
    planes_Tmass: Dict[int, float] = {}
    planes_P: Dict[int, float] = {}
    planes_A: Dict[int, float] = {}
    planes_mdot: Dict[int, float] = {}
    planes_rho: Dict[int, float] = {}
    planes_mu: Dict[int, float] = {}
    planes_k: Dict[int, float] = {}

    # pasma (wall_band_i)
    bands_Awet: Dict[int, float] = {}
    bands_Q: Dict[int, float] = {}
    bands_Tw: Dict[int, float] = {}

    A_env_single: Optional[float] = None

    i = 0
    while i < n:
        line = lines[i]

        # --- AREA [m^2] (przekroje i zwilżona powierzchnia pasm) ---
        if "Area" in line and "[m^2]" in line:
            j = i + 1
            while j < n and not RE_DASH.search(lines[j]):  # idź do linii '----'
                j += 1
            if j < n:
                table, i2 = _collect_rows(lines, j)
                table = _norm_table(table)
                # envelope_xy (czasem występuje)
                if "envelope_xy" in table:
                    A_env_single = table["envelope_xy"]
                # ziso_i -> pole przekroju
                for k, v in table.items():
                    if k.startswith("ziso_"):
                        try: planes_A[int(k.split("_")[1])] = v
                        except Exception: pass
                # wall_band_i -> zwilżona powierzchnia
                for k, v in table.items():
                    if k.startswith("wall_band_"):
                        try: bands_Awet[int(k.split("_")[2])] = v
                        except Exception: pass
                i = i2
                continue

        # --- TEMPERATURE [K] (T_mass na ziso_* i T_wall na wall_band_*) ---
        # Łapiemy zarówno "Area-Weighted Average" jak i "Mass-Weighted Average",
        # a słowo "Temperature" może być w tej samej linii lub w kolejnej.
        if ("Average" in line) and (
            "Temperature" in line or (i + 1 < n and "Temperature" in lines[i + 1])
        ):
            # przejdź do linii '----' oddzielającej nagłówek od tabeli
            j = i + 1
            while j < n and not RE_DASH.search(lines[j]):
                j += 1
            if j < n:
                table, i2 = _collect_rows(lines, j)
                table = _norm_table(table)
                # ziso_i → T_mass
                for k, v in table.items():
                    if k.startswith("ziso_"):
                        try: planes_Tmass[int(k.split("_")[1])] = v
                        except Exception: pass
                # wall_band_i → T_wall  (TO BYŁO KLUCZOWE)
                for k, v in table.items():
                    if k.startswith("wall_band_"):
                        try: bands_Tw[int(k.split("_")[2])] = v
                        except Exception: pass
                i = i2
                continue

        # --- STATIC PRESSURE [Pa] ---
        if ("Area-Weighted Average" in line) and (i + 1 < n and "Static Pressure" in lines[i + 1]):
            j = i + 1
            while j < n and not RE_DASH.search(lines[j]):
                j += 1
            if j < n:
                table, i2 = _collect_rows(lines, j)
                table = _norm_table(table)
                for k, v in table.items():
                    if k.startswith("ziso_"):
                        try: planes_P[int(k.split("_")[1])] = v
                        except Exception: pass
                i = i2
                continue

        # --- MASS FLOW RATE [kg/s] ---
        if "Mass Flow Rate" in line and "[kg/s]" in line:
            j = i + 1
            while j < n and not RE_DASH.search(lines[j]):
                j += 1
            if j < n:
                table, i2 = _collect_rows(lines, j)
                table = _norm_table(table)
                for k, v in table.items():
                    if k.startswith("ziso_"):
                        try: planes_mdot[int(k.split("_")[1])] = v
                        except Exception: pass
                i = i2
                continue

        # --- DENSITY [kg/m^3] ---
        if ("Area-Weighted Average" in line) and (i + 1 < n and "Density" in lines[i + 1]):
            j = i + 1
            while j < n and not RE_DASH.search(lines[j]):
                j += 1
            if j < n:
                table, i2 = _collect_rows(lines, j)
                table = _norm_table(table)
                for k, v in table.items():
                    if k.startswith("ziso_"):
                        try: planes_rho[int(k.split("_")[1])] = v
                        except Exception: pass
                i = i2
                continue

        # --- VISCOSITY [Pa·s] ---
        if ("Area-Weighted Average" in line) and (i + 1 < n and "Viscosity" in lines[i + 1]):
            j = i + 1
            while j < n and not RE_DASH.search(lines[j]):
                j += 1
            if j < n:
                table, i2 = _collect_rows(lines, j)
                table = _norm_table(table)
                for k, v in table.items():
                    if k.startswith("ziso_"):
                        try: planes_mu[int(k.split("_")[1])] = v
                        except Exception: pass
                i = i2
                continue

        # --- THERMAL CONDUCTIVITY [W/m-K] ---
        if ("Area-Weighted Average" in line) and (i + 1 < n and "Thermal Conductivity" in lines[i + 1]):
            j = i + 1
            while j < n and not RE_DASH.search(lines[j]):
                j += 1
            if j < n:
                table, i2 = _collect_rows(lines, j)
                table = _norm_table(table)
                for k, v in table.items():
                    if k.startswith("ziso_"):
                        try: planes_k[int(k.split("_")[1])] = v
                        except Exception: pass
                i = i2
                continue

        # --- HEAT FLUX / HEAT TRANSFER RATE (Q pasm) ---
        if any(kw in line for kw in ["Total Surface Heat Flux", "Total Heat Transfer Rate", "Heat Transfer Rate", "Surface Heat Flux"]):
            j = i + 1
            while j < n and not RE_DASH.search(lines[j]):
                j += 1
            if j < n:
                table, i2 = _collect_rows(lines, j)
                table = _norm_table(table)
                for k, v in table.items():
                    if k.startswith("wall_band_"):
                        try: bands_Q[int(k.split("_")[2])] = v
                        except Exception: pass
                i = i2
                continue

        i += 1

    if verbose:
        print(f"[PARSE] planes: T={len(planes_Tmass)} P={len(planes_P)} A={len(planes_A)} "
              f"mdot={len(planes_mdot)} rho={len(planes_rho)} mu={len(planes_mu)} k={len(planes_k)}")
        print(f"[PARSE] bands : Awet={len(bands_Awet)} Q={len(bands_Q)} Tw={len(bands_Tw)}; A_env={A_env_single}")

    return SRPData(
        planes_Tmass=planes_Tmass, planes_P=planes_P, planes_A=planes_A,
        planes_mdot=planes_mdot, planes_rho=planes_rho, planes_mu=planes_mu, planes_k=planes_k,
        A_env_single=A_env_single,
        bands_Awet=bands_Awet, bands_Q=bands_Q, bands_Tw=bands_Tw,
    )
