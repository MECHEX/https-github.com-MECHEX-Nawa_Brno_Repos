# Transient_Repo — Transient Simulation Post-Processing

## Purpose

Processes **time-resolved** ANSYS Fluent simulations of gyroid heat exchangers. Each timestep
produces a pair of SRP files (Fluid1, Fluid2); this repo indexes them by time, computes thermal-
hydraulic profiles, and generates overlay/mean plots and cross-case comparisons.

---

## Architecture

```
config.py          ← master config: CASES, PLOT_JOBS, COMPARE_JOBS, FLUID_CFG
     │
     ▼
config_resolver.py ← validates config, flattens CASES into part_catalog,
                     merges SRP_DEFAULTS into each part's metadata
     │
     ▼
main.py            ← CLI entry point
  ├── indexes SRP files per (case, part, fluid, time)
  ├── for each active PLOT_JOB  → plotting.py (overlay + mean plots)
  └── for each active COMPARE_JOB → compare_runner.py → compare.py

convergence.py     ← standalone script: mean_h and mean_f vs. time per case
```

**Supporting modules:**

| Module | Role |
|--------|------|
| `srp_parser.py` | Parses plain-text SRP files → `SRPData` namedtuple |
| `compute.py` | `compute_band_table()` — h, Nu, Re, f per band; `compute_global_means()` |
| `tpms_utils.py` | Filename utilities: `extract_step()`, `fluid_from_name()`, `collect_all_data()` |
| `constants.py` | Column name constants, axis candidates, ylabel/prefix maps |
| `style.py` | Deterministic color/marker assignment by label hash |

---

## Simulation Dataset (`CASES` in `config.py`)

| Case ID | Geometry | Fluid Pair | Turbulence | Parts | Time Range | Notes |
|---------|----------|-----------|-----------|-------|-----------|-------|
| `grad_aw_laminar` | GRAD | Air/Water | Laminar | part1, part2, part3 | 0–11 s | |
| `grad_aw_kw_earsm` | GRAD | Air/Water | k-ω EARSM | part4 | 0–10 s | |
| `grad_aw_laminar_part10` | GRAD | Air/Water | Laminar | part10 | 0–0.68 s | short export, `t_interpretation="local"` |
| `grad_aw_laminar_part11` | GRAD | Air/Water | Laminar | part11 | 0.02–20.0 s | long Mesh_003_v4 transient; reference for M003_v4 steady |
| `grad_ao_laminar` | GRAD | Air/Oil | Laminar | part5, part6 | 0–20 s | |
| `grad_ao_kw_earsm` | GRAD | Air/Oil | k-ω EARSM | part7 | 0–10 s | |
| `grad_ao_kw_earsm_restart` | GRAD | Air/Oil | k-ω EARSM | part8 | 9.91–19.91 s | |
| `grad_ao_kw_earsm_restart_lowres` | GRAD | Air/Oil | k-ω EARSM | part9 | 9.91–19.91 s | |
| `uni10_003` | UNI10 | Air/Water | Laminar | main | 0–20 s | |
| `uni10_005` | UNI10 | Air/Water | Laminar | main | 0–21.28 s | |
| `uni10_005_hfe` | UNI10 | Air/HFE | Laminar | main | 0–19.98 s | HFE cold side, `t_interpretation="local"` |
| `uni10_005_oil` | UNI10 | Air/Oil | Laminar | main | 0–19.99 s | oil cold side, `t_interpretation="local"` |
| `uni10_007` | UNI10 | Air/Water | Laminar | main | 0–20.14 s | |

All cases above currently carry `"active": True` in `config.py`. Case-level `active` only controls
auto-indexing when no active job references the case (see `"active"` note under `CASES` below).

---

## SRP File Naming & Time Extraction

SRP files reside in `FluentTransientData/<source_dir>/`. Two naming schemes are supported:

```
<prefix>_S04000_Fluid1.srp   →  step = 4000, t_local = step × dt_sim_s
<prefix>_T2.0000_Fluid1.srp  →  time token = 2.0 s  (global or local per t_interpretation)
```

Priority order is set per-part via `time_source_priority` (default `("T", "S")`).

`t_interpretation` (default `"global"`) controls how the `T` token is interpreted:
- `"global"` — subtract `t_start_s` to get local time
- `"local"` — use as-is
- `"auto"` — detect based on magnitude

---

## Configuration Objects

### `CASES` — simulation families
```python
CASES = {
    "uni10_003": {
        "active": True,
        "geometry": "uni10",         # "grad" or "uni10"
        "run": "003",                # run identifier
        "description": "...",
        "parts": {
            "main": {
                "source_dir": "uni10_003",   # dir under FluentTransientData/
                "t_start_s": 0.0,
                "t_end_s": 20.0,
                "dt_sim_s": 0.0005,          # simulation timestep [s]
            },
        },
    },
}
```

`"active": False` on a case does NOT prevent it from being indexed if a COMPARE_JOB references
its parts. Inactivity only suppresses auto-indexing when no job references it.

### `PLOT_JOBS` — per-case time-evolution plots
```python
{
    "name": "job_uni10_003",
    "active": True,
    "members": [{"case_id": "uni10_003"}],     # can span multiple cases
    "fluids": ["Fluid1", "Fluid2"],
    "plots": ["overlay", "mean"],              # "overlay" = all timesteps, "mean" = mean vs time
    "metrics": ["h", "f"],
    "overlay_mode": "mean_std",                # "all" or "mean_std"
}
```

Output → `TransientFigs/plots/<job_name>/`

### `COMPARE_JOBS` — cross-case comparisons
```python
{
    "name": "compare_uni10_with_steady",
    "active": False,                          # set True to run
    "plots": ["overlay", "mean"],
    "metrics": ["h", "f"],
    "series": [
        {
            "label": "UNI10 003",
            "case_id": "uni10_003",
            "parts": ["main"],
            "fluid": "Fluid1",
            "t0_s": 18.0, "t1_s": 19.0,     # time window for averaging
        },
        ...
    ],
    "steady": {
        "enabled": True,
        "base_csv_dir": r"..\Steady_Repo\DataProcessed\csv",
        "cases": {
            "Fluid1": [{"label": "UNI10 003", "file": "Sim_Data_Fluid1_GUNI_003.csv"}],
        },
    },
    "style_map": {"UNI10 003": {"color": "#17becf", "ls": "-", "marker": "o"}},
    "overlay": {"time_avg": {"mode": "mean", "weights": "auto"}, "shade": "std"},
    "mean": {"time_mode": "aligned", "ma_windows": [2], ...},
    "fig": {"dpi": 450, "overlay": {"figsize": (14.0, 5.2)}, "mean": {"figsize": (14.0, 5.2)}},
}
```

Output → `TransientFigs/Compare/<job_name>/`

### `FLUID_CFG` — axis geometry per fluid
```python
FLUID_CFG = {
    "Fluid1": {"axis": "z", "min": -0.02959, "max": 0.02920, "step": 0.001959666667},
    "Fluid2": {"axis": "y", "min": 0.00950, "max": -0.00953, "step": 0.0009515},
}
```
Must match `Steady_Repo/constants.py → AXIS_RANGES`.

---

## Representative Time Windows

Time windows used when averaging transient results for comparison plots:

```python
# GRAD parts (pick a 1-second window in the quasi-steady period)
GRAD_COMPARE_WINDOWS_S = {
    "part1": (3.0, 4.0),   "part2": (8.0, 9.0),   "part3": (10.0, 11.0),
    "part4": (4.0, 5.0),   "part5": (5.0, 6.0),   "part6": (15.0, 16.0),
    "part7": (8.0, 9.0),   "part8": (18.0, 19.0),  "part9": (18.0, 19.0),
    "part11": (18.0, 19.0),
}

# UNI10 runs (last second of simulation — converged region)
UNI_COMPARE_WINDOWS_S = {
    "uni10_003": (18.0, 19.0),
    "uni10_005": (18.0, 19.0),
    "uni10_007": (18.0, 19.0),
}
```

See also `convergence.py` for a diagnostic plot of mean_h/mean_f vs. time to verify convergence.

---

## Active Jobs (Current State)

State reflects `config.py` (verify there if in doubt — the config is the source of truth).

| Job | Type | Active | Output |
|-----|------|--------|--------|
| `job_grad_part1..part9` | PLOT_JOB | ✗ | `plots/job_grad_part*/` |
| `job_grad_part10` | PLOT_JOB | ✓ | `plots/job_grad_part10/` |
| `job_grad_part11` | PLOT_JOB | ✓ | `plots/job_grad_part11/` |
| `job_uni10_003/005/007` | PLOT_JOB | ✓ | `plots/job_uni10_*/` |
| `compare_grad_P1_P9` | COMPARE_JOB | ✗ | `Compare/compare_grad_P1_P9/` |
| `compare_grad_uni10_001` | COMPARE_JOB | ✓ | `Compare/compare_grad_uni10_001/` |
| `compare_uni10_with_steady` | COMPARE_JOB | ✗ | `Compare/compare_uni10_with_steady/` |
| `compare_grad_vs_uni10` | COMPARE_JOB | ✓ | `Compare/compare_grad_vs_uni10/` |
| `compare_uni10_005_coldfluid` | COMPARE_JOB | ✓ | `Compare/compare_uni10_005_coldfluid/` |

Note: some inactive jobs still have output on disk from earlier runs (e.g. `compare_grad_P1_P9`,
`compare_uni10_with_steady`). Presence of an output directory does not imply the job is currently active.

To activate a job: set `"active": True` in the corresponding entry in `config.py`.

---

## Running

```bash
cd Transient_Repo

# Main pipeline (PLOT_JOBS + COMPARE_JOBS)
python main.py
python main.py --in-dir /path/to/FluentTransientData --out-dir /path/to/output --verbose

# Temporal convergence analysis
python convergence.py
python convergence.py --cases uni10_003 uni10_005 --out-dir /path/to/output
```

**Path note (reproducibility):** `config.py` defines `DEFAULT_BASE_DATA_DIR` / `DEFAULT_OUT_DIR`
pointing at a `My Drive\...` location, whereas this working checkout lives under
`Documents\PK and PR\Brno_NAWA\Repo\...`. The two are separate copies. Do **not** assume the
defaults match your checkout — pass `--in-dir` / `--out-dir` explicitly, or confirm which location
is canonical before relying on the hard-coded defaults. (Open question flagged for the repo owner.)

---

## Adding a New Simulation Case

1. Copy SRP files to `FluentTransientData/<new_dir>/`
2. Add an entry to `CASES` in `config.py`:
   ```python
   "my_new_case": {
       "active": True,
       "geometry": "uni10",   # or "grad"
       "run": "010",
       "description": "...",
       "parts": {
           "main": {"source_dir": "my_new_dir", "t_start_s": 0.0, "t_end_s": 20.0, "dt_sim_s": 0.0005},
       },
   }
   ```
3. Add a `PLOT_JOB` entry referencing the new case, or reference it in an existing/new `COMPARE_JOB`.
4. Run `python main.py`.

---

## Refactoring Status

The config was refactored from a flat `PARTS` dict to a case-centric `CASES + parts` hierarchy.
`config_resolver.py` validates the config and builds a flat `part_catalog` at runtime (keyed as
`"case_id::part_id"`), maintaining backward compatibility with all plotting functions.
The old `README.md` described the in-progress refactoring plan; this CLAUDE.md supersedes it.
