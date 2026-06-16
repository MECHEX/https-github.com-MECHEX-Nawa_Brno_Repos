# NAWA Brno — CFD Post-Processing Repository

## Research Context

This repository contains post-processing code for **ANSYS Fluent** CFD simulations of heat
exchangers based on **TPMS (Triply Periodic Minimal Surface) gyroid** structures, developed
as part of a NAWA academic exchange collaboration between Politechnika Krakowska (PL) and
a Brno university (CZ).

### Geometry Variants

| ID | Name | Description |
|----|------|-------------|
| **GRAD** | Graded Gyroid | Variable cell size along the flow direction: small cells at inlet → large cells at outlet |
| **UNI10** | Uniform Gyroid | Uniform cell size throughout the volume; cell size = 10 mm |

Both geometries form a compact counter-flow heat exchanger where two fluid streams (Fluid1,
Fluid2) flow through separate, interleaved gyroid channels separated by the solid gyroid wall.

### Fluid Pairs Simulated

| Label | Fluid1 | Fluid2 |
|-------|--------|--------|
| Air/Water (AW) | Air (hot side) | Water (cold side) |
| Air/Oil (AO) | Air (hot side) | Oil (cold side) |

Fluid1 is resolved along the **z-axis**; Fluid2 along the **y-axis** (see `FLUID_CFG` / `AXIS_RANGES`
in each sub-repo's `constants.py` or `config.py`).

### Turbulence Models

- **Laminar** — no turbulence model
- **k-ω EARSM** — Explicit Algebraic Reynolds Stress Model

---

## Repository Map

```
Nawa_Brno_Repos/
├── CLAUDE.md                   ← this file
├── Steady_Repo/                ← steady-state mesh study (see Steady_Repo/CLAUDE.md)
│   ├── CLAUDE.md
│   ├── main.py
│   ├── compute.py
│   ├── plotting.py
│   ├── srp_parser.py
│   ├── constants.py
│   ├── FluentRaport/           ← input SRP files (one per mesh/case)
│   └── DataProcessed/          ← CSV outputs + PNG plots
└── Transient_Repo/             ← transient simulations (see Transient_Repo/CLAUDE.md)
    ├── CLAUDE.md
    ├── config.py               ← master configuration (CASES, PLOT_JOBS, COMPARE_JOBS)
    ├── main.py
    ├── compare.py / compare_runner.py
    ├── compute.py
    ├── plotting.py
    ├── srp_parser.py
    ├── constants.py
    ├── config_resolver.py
    ├── tpms_utils.py
    ├── convergence.py          ← temporal convergence analysis (mean_h, mean_f vs time)
    ├── FluentTransientData/    ← input SRP files (one directory per simulation part)
    └── TransientFigs/          ← CSV outputs + PNG plots
```

---

## SRP File Format

SRP (Solver Report) files are plain-text output from ANSYS Fluent containing area-/mass-weighted
averages sampled at **iso-planes** along the flow axis and on **wall bands** between adjacent planes.

Key sections extracted by `srp_parser.py`:

| Section | What it contains |
|---------|-----------------|
| `[AREA]` / `[FLOW_AREA]` | Cross-sectional area A_flow at each iso-plane |
| `[TEMPERATURE]` | Mass-weighted bulk temperature T_mass at each plane |
| `[PRESSURE]` | Area-averaged pressure P_area at each plane |
| `[MASS_FLOW]` | Mass flow rate ṁ at each plane |
| `[DENSITY]` | Fluid density ρ at each plane |
| `[VISCOSITY]` | Dynamic viscosity μ at each plane |
| `[CONDUCTIVITY]` | Thermal conductivity k at each plane |
| `[AREA_WALL]` | Wetted wall area A_wet for each band |
| `[HEAT_WALL]` | Heat transfer rate Q for each band |
| `[TEMPERATURE_WALL]` | Wall temperature T_wall for each band |

---

## Key Computed Quantities

All computed in `compute.py → compute_band_table()`:

### Hydraulic Diameter (gyroid method)
```
Dh = 4 × A_flow_mid × dz_local / A_wet
```
Fallback (area-equivalent): `Dh = sqrt(4 × A_flow / π)`

### Heat Transfer Coefficient
```
h = |Q_band| / (A_wet × LMTD_wall)
```
where `LMTD_wall` = log-mean temperature difference between wall and fluid (inlet/outlet of band).

### Reynolds Number
```
Re = ρ_mid × U × Dh / μ_mid
where U = ṁ_mid / (ρ_mid × A_flow_mid)
```

### Nusselt Number
```
Nu = h × Dh / k_mid
```

### Fanning Friction Factor
```
f_Fanning = (Δp_band / dz_local) × Dh / (2 × ρ_mid × U²)
```

### Performance Metrics (steady, aggregate)
- `Q/P_pump` — heat transferred per unit pump power [W/W]
- `UA/P_pump` — overall conductance per unit pump power [1/K per W]
- `Nu/f^(1/3)` — thermal-hydraulic performance factor

---

## Recommended Meshes (Mesh Convergence Results)

Full analysis: `Steady_Repo/mesh_convergence.py` → `Steady_Repo/DataProcessed/convergence/`

| Geometry | Recommended Mesh | Cells | Basis |
|----------|-----------------|-------|-------|
| **UNI10** | **GUNI_007** | 6.25M | GCI < 2.3% vs GUNI_008 (16.7M); clean convergence pair |
| **GRAD** | **M006** (Laminar) or **M007** (k-ω EARSM) | 881K | Series M001–M004 confounded by simultaneous BL setting changes |

**Critical note for GRAD:** The series M001→M004 is NOT a clean mesh convergence study —
boundary layer type and layer count changed simultaneously with cell count. Fluid2_h shows
a +16% jump from M001/M002 to M003, which is an artifact of the BL setup, not mesh density.
The turbulence model sensitivity at 881K is low: F1_h varies <0.7%, F1_f varies ~5.3%.

**Critical note for UNI10:** Only GUNI_007 and GUNI_008 form a clean convergence pair
(identical surface mesh settings). The sequence GUNI_003→005→007 shows non-monotonic
behaviour due to changing surface mesh quality between runs.

---

## Naming Conventions

| Pattern | Meaning |
|---------|---------|
| `Fluid1` | First fluid (air side, z-axis) |
| `Fluid2` | Second fluid (water/oil side, y-axis) |
| `M001`–`M007` | GRAD geometry steady-state mesh IDs |
| `GUNI_001/003/005/007/008` | UNI10 geometry steady-state mesh IDs |
| `part1`–`part9` | GRAD transient simulation segments |
| `uni10_003/005/007` | UNI10 transient simulation runs |
| `_AW_` | Air/Water fluid pair |
| `_AO_` | Air/Oil fluid pair |
| `_Lam_` | Laminar turbulence model |
| `_kw_` | k-ω EARSM turbulence model |
| `SS` | Steady-state reference |

---

## Compute Environment

Simulations were run on **ANSYS Fluent 2024 R2** using 130 cores (AMD EPYC 9654, 96-core × 2)
at ICM (Interdisciplinary Centre for Mathematical and Computational Modelling, University of Warsaw).
