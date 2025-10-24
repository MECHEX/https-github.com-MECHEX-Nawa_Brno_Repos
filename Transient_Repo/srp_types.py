# -*- coding: utf-8 -*-
"""
srp_types.py — proste typy dla przejrzystości.
"""
from typing import Dict, Optional, NamedTuple

class SRPData(NamedTuple):
    # z-planes (ziso_i)
    planes_Tmass: Dict[int, float]       # K
    planes_P: Dict[int, float]           # Pa
    planes_A: Dict[int, float]           # m^2 (flow area)
    planes_mdot: Dict[int, float]        # kg/s
    planes_rho: Dict[int, float]         # kg/m^3
    planes_mu: Dict[int, float]          # Pa·s
    planes_k: Dict[int, float]           # W/mK

    # jeśli SRP ma pojedynczą 'envelope_xy' (przekrój obrysu)
    A_env_single: Optional[float]        # m^2

    # wall bands (band_i)
    bands_Awet: Dict[int, float]         # m^2 powierzchni zwilżonej w paśmie
    bands_Q: Dict[int, float]            # W (calka strumienia ciepła)
    bands_Tw: Dict[int, float]           # K
