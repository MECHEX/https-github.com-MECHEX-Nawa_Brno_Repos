# Opis figur — Transient_Repo (symulacje niestacjonarne TPMS gyroid)

Wszystkie wyniki trafiają do `TransientFigs/`. Ten plik opisuje strukturę katalogów,
konwencję nazw plików i znaczenie elementów wykresów. Źródłem prawdy o tym, które joby
są aktywne i jakie serie zawierają, jest `config.py` (`PLOT_JOBS`, `COMPARE_JOBS`).

---

## Struktura katalogów `TransientFigs/`

| Katalog | Zawartość | Generowany przez |
|---------|-----------|------------------|
| `plots/<job>/` | Wykresy per-case: ewolucja czasowa metryk dla jednego PLOT_JOB | `main.py` → `plotting.py` |
| `Compare/<job>/` | Porównania międzyprzypadkowe dla jednego COMPARE_JOB | `main.py` → `compare_runner.py` → `compare.py` |
| `convergence/` | Zbieżność temporalna (mean_h / mean_f vs czas) | `convergence.py` (osobny skrypt) |
| `csv/<job>/` | Tabele pośrednie `summary_means` (uśrednienia przestrzenne per krok) | `main.py` |

---

## PLOT_JOBS — `plots/<job>/`

Wykresy ewolucji czasowej dla pojedynczego przypadku (lub kilku scalonych w jeden job).

### Konwencja nazw

```
mean_<metric>__<job>__<fluid>__<dt>.png
overlay_<metric>_meanstd__<job>__<fluid>__<dt>.png
```

- `<metric>` ∈ {`h`, `f`}  — współczynnik konwekcji / współczynnik tarcia Fanninga
- `<job>` — nazwa PLOT_JOB z `config.py` (np. `job_uni10_003`)
- `<fluid>` ∈ {`F1`, `F2`}  — Fluid1 (powietrze, oś z) / Fluid2 (czynnik zimny, oś y)
- `<dt>` — krok symulacyjny w notacji naukowej (np. `5e-04s`)

### Typy wykresów

| Prefiks | Typ | Osie | Cel |
|---------|-----|------|-----|
| `mean_*` | Linia | X = czas [s], Y = przestrzennie uśrednione h lub f | Zbieżność temporalna wielkości uśrednionej po kanale |
| `overlay_*_meanstd` | Wstęga | X = pozycja wzdłuż kanału, Y = h lub f | Profil przestrzenny; wstęga = ±std po krokach czasu (tryb `overlay_mode="mean_std"`) |

`overlay_mode` (w PLOT_JOB) przełącza `all` (wszystkie kroki na jednym wykresie) vs
`mean_std` (średnia + wstęga odchylenia standardowego).

---

## COMPARE_JOBS — `Compare/<job>/`

Porównania międzyprzypadkowe. Każdy job definiuje listę `series` (transient) plus opcjonalny
blok `steady` z referencjami stacjonarnymi.

### Konwencja nazw

```
cmp_overlay_<metric>_<fluid>__<job>.png
cmp_mean_<metric>_<fluid>__aligned__<job>.png
```

- `<metric>` ∈ {`h`, `f`}, `<fluid>` ∈ {`F1`, `F2`}
- `aligned` — tryb `mean.time_mode="aligned"` (osie czasu przypadków wyrównane do wspólnej osi)

### Typy wykresów

| Prefiks | Typ | Osie | Cel |
|---------|-----|------|-----|
| `cmp_overlay_*` | Profile przestrzenne wielu przypadków | X = pozycja wzdłuż kanału, Y = metryka | Porównanie rozkładów przestrzennych w reprezentatywnym oknie czasu |
| `cmp_mean_*_aligned` | Szeregi czasowe wielu przypadków | X = czas [s], Y = uśredniona metryka | Porównanie ewolucji temporalnej między przypadkami |

### Elementy wykresów compare

| Element | Znaczenie |
|---------|-----------|
| Linie ciągłe/przerywane kolorowe | Przebiegi transient poszczególnych `series` (styl z `style_map`) |
| Linie kropkowane z sufiksem **`SS`** | Referencje stacjonarne (steady-state) z `Steady_Repo/DataProcessed/csv` |
| Wstęga (`shade="std"`) | ±odchylenie standardowe uśrednienia czasowego (overlay) |

Etykiety `*_SS` (np. `M006 SS`, `GUNI 007 SS`) to linie poziome/profile z CSV steady —
punkt odniesienia „do czego dąży transient". Okna czasu uśredniania serii transient pochodzą
z `GRAD_COMPARE_WINDOWS_S` / `UNI_COMPARE_WINDOWS_S` w `config.py`.

### Aktualne COMPARE_JOBS (stan wg `config.py`)

| Job | Aktywny | Co porównuje |
|-----|---------|--------------|
| `compare_grad_P1_P9` | ✗ | Wszystkie segmenty GRAD (part1–part10) + steady M006/M007 |
| `compare_grad_uni10_001` | ✓ | GRAD (part1–10) vs UNI10 (003/005/007 + HFE + Oil) + steady |
| `compare_uni10_with_steady` | ✗ | UNI10 003/005/007 transient vs własne referencje GUNI steady |
| `compare_grad_vs_uni10` | ✓ | GRAD AW (part1–4) vs UNI10 AW (003/005/007) + steady M006/M007, GUNI 003/005/007 |
| `compare_uni10_005_coldfluid` | ✓ | UNI10 bieg 005: **woda vs olej vs HFE** (efekt czynnika zimnego); bez referencji steady |

> **compare_uni10_005_coldfluid — efekt czynnika strony zimnej:** ta sama geometria, siatka i bieg
> (UNI10 005), zmienny wyłącznie czynnik zimny (Fluid2). Różnice na Fluid2 wynikają czysto z
> właściwości fizycznych (Pr, ρ, μ, k); Fluid1 (powietrze) pokazuje sprzężenie przez ściankę.
> Steady wyłączony celowo — tylko woda ma odpowiednik stacjonarny (GUNI_005), więc pojedyncza
> linia steady byłaby asymetryczna. Obserwacja: h_F2 wody (~3700 W/m²K) jest ~3× wyższe niż
> oleju/HFE (~1100–1200 W/m²K) — zgodnie z wyższą przewodnością i Pr wody.

> **Uwaga interpretacyjna (compare_grad_vs_uni10):** GRAD i UNI10 pracują w różnych punktach
> Re (F1 ≈ 341 vs 218 — patrz `Steady_Repo` fig_F). Bezpośrednie porównanie h/f bez uwzględnienia
> różnicy Re jest mylące; interpretować łącznie z Re danego przypadku.

### Mesh_003_v4 / part11 transient reference

`job_grad_part11` przetwarza dlugi transient Air/Water Laminar dla GRAD `Mesh_003_v4`
(`part11`, 0.02-20.0 s, dt=5e-4). Wyniki sa uzywane jako punkt referencyjny
`M003_v4 transient` na wykresach zbieznosci steady (`fig_G3_combined_conv_v2-v5`) i sa
liczone z okna 18-19 s. Krotki `part10` pozostaje zapisany w rejestrze danych, ale do
aktualnych wykresow zbieznosci preferowany jest `part11`.

---

## Zbieżność temporalna — `convergence/`

Skrypt `convergence.py` (uruchamiany osobno). Diagnostyka „czy przypadek zbiegł w czasie".

| Plik | Zawartość |
|------|-----------|
| `conv_h_F1.png` / `conv_f_F1.png` | mean_h / mean_f (Fluid1) vs czas dla wybranych przypadków |
| `conv_h_F1.csv` / `conv_f_F1.csv` | Dane liczbowe do powyższych wykresów |

Służy do potwierdzenia, że okno uśredniania w `*_COMPARE_WINDOWS_S` leży w obszarze quasi-ustalonym.

---

*Wykresy PLOT_JOB/COMPARE_JOB generowane przez `main.py`; zbieżność temporalna przez `convergence.py`.*
*Dane wejściowe: `FluentTransientData/<source_dir>/*.srp`; referencje steady: `..\Steady_Repo\DataProcessed\csv`.*
