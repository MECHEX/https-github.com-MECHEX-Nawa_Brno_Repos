# Plan przygotowania wyników do artykułu — TPMS Gyroid Heat Exchanger

Wygenerowano jako wykonanie promptu „zaplanuj przygotowanie wyników do artykułu"
(patrz `research_log.md`). Rozdział materiału na: **Grid Independence**, **V&V**, **Wyniki**.
Data: 2026-07-07.

> **Uwaga metodyczna:** *Weryfikacja* = „czy równania rozwiązano poprawnie" (siatka, krok czasowy,
> residua, bilans energii). *Walidacja* = „czy rozwiązano właściwe równania" (eksperyment/korelacje).
> Grid independence jest formalnie częścią weryfikacji, ale w artykule prowadzimy ją jako osobną
> podsekcję (tak jak chciał autor).

---

## 0. Ustalenia z analizy danych (nowe, policzone teraz)

| Ustalenie | Wynik | Konsekwencja dla artykułu |
|---|---|---|
| **Bilans energii** Q_air vs Q_cold (sum \|Q_band\|) | GRAD M006: −0.05% · M007: +0.03% · UNI10 GUNI_007: −0.05% | **DOMYKA SIĘ ~0.05%** → mocny wynik weryfikacyjny; brak #1 ROZWIĄZANY |
| **Wyjaśnienie dT_F2 ≈ 0.08 K** | Q≈7 W, woda ma ogromne ṁ·cp → prawie się nie grzeje | NIE jest błędem; fizycznie spójne z bilansem — do opisania w tekście |
| **Punkt pracy (steady AW)** | T_in air ≈ 298 K, T_in „water" = 363 K; Q ≈ 7 W/wymiennik | Woda wchodzi gorąca i **grzeje** powietrze |
| **Niespójność hot/cold** | W BC steady woda (Fluid2) jest stroną GORĄCĄ (363 K), powietrze zimną — odwrotnie niż „air hot side" w `CLAUDE.md` | **DO ROZSTRZYGNIĘCIA**: poprawić opis albo wyjaśnić konwencję |
| **Re (mean)** | GRAD: air 341 / water 314; UNI10: air 218 / water 221 | Reżim laminarny/przejściowy → uzasadnia model Laminar |

---

## A. GRID INDEPENDENCE STUDY

### Co mamy
- **GCI (Celik/Roache, p=2)** — `DataProcessed/convergence/gci_results.csv`: h_F1, Nu_F1, f_F1 dla UNI10;
  czysta para GUNI_007/008 → h_F1 GCI = 1.06%, Nu 1.08%, f 2.27%.
- Figury: G1/G2 (h,f vs N), G3 (% odchylenia), G1_v2/G2_v2/G3_v2 (dT,dp), G4_v1–v3 (dp/dT/Q), G5–G7.
- Zalecane siatki: **GRAD M006/M007 @881K**, **UNI10 GUNI_007 @6.25M**.
- Uczciwa dyskusja: seria GRAD M001–M004 confounded (BL + N zmieniane jednocześnie).

### Braki / do zrobienia
| # | Brak | Nakład | Źródło |
|---|---|---|---|
| A1 | GCI tylko dla Fluid1 — brak dla **Fluid2** | mały | rozszerzyć `mesh_convergence.py` (dane już są w `UNI` dict) |
| A2 | GRAD bez klasycznego GCI — potrzebne **jawne uzasadnienie** w tekście (nie liczba) | mały | opis + odwołanie do sensitivity M005/006/007 |
| A3 | Tabela zbieżności „ready-to-print" (N, h, Nu, f, GCI% per geometria) | mały | z `convergence_summary.csv` + `gci_results.csv` |

### Do artykułu
- 1 figura zbieżności (rekomendacja: **G3** lub **G7** — najczytelniejsze, % odchylenia) + tabela GCI.
- Pozostałe G* → supplementary.

---

## B. VERIFICATION & VALIDATION

### B1. Verification (poza siatką)

| Element | Status | Co zrobić |
|---|---|---|
| **Bilans energii** | ✅ policzony (±0.05%) | Zrobić mini-tabelę/figurę + zdanie w tekście |
| **Transient → steady** | ✅ `fig_G8` (h,dT ±2%; f/dp EARSM −5%/−3.3%) | Gotowe; opis interpretacyjny |
| **Zbieżność w czasie** | ✅ `convergence.py` (mean_h/f vs t) | Wybrać 1 reprezentatywny przebieg do figury |
| **Residua / iterative convergence** | ❌ brak w repo | **Eksport z Fluent** (monitory residuów) — must-have |
| **y+ / rozdzielczość przyścienna** | ❌ brak | **Eksport z Fluent** (kontur/statystyka y+) — kluczowe dla EARSM |
| **Time-step independence** | ⚠️ dane są (dt=1e-4/2.5e-4/5e-4), brak studium | Porównać part3(1e-4) vs reszta, lub dedykowany krótki przebieg |

### B2. Validation

| Element | Status | Uwaga |
|---|---|---|
| **Korelacje literaturowe** | ✅ `literature_comparison.png`, fig_J (Zhukauskas, Briggs-Young), fig_K (Hausen) | Solidne dla korelacji |
| **Eksperyment** | ❌ brak | **Decyzja autora**: pozostać przy „validation vs correlations" (uczciwe ograniczenie) albo pozyskać dane z Brna |

### Do artykułu
- Verification: tabela (bilans energii + GCI + residua + y+) + `fig_G8`.
- Validation: `literature_comparison` do głównego tekstu; fig_J/K do głównego lub supplementary.

---

## C. WYNIKI (Results & Discussion)

### C1. Porównanie geometrii (steady, Air/Water) — rdzeń artykułu
Mamy komplet:
- **fig_A** — profile lokalne (kluczowe: monotoniczny wzrost Re(z) w GRAD = dowód działania gradientu).
- **fig_C** — Δp skumulowany (UNI10 ~20% wyższy Δp).
- **fig_B / fig_B2** — wskaźniki termohydrauliczne (B2 = h/Δp^⅓ niezależny od Dh — uczciwsze niż Nu/f^⅓).
- **fig_D** — mapa Q/Ppump i UA/Ppump.
- **fig_E** — radar (6 metryk).
- **fig_F** — bar (zestawienie tabelaryczne graficzne).
- **fig_H/I/L** — rozkład ciepła wzdłuż kanału.

### C2. Wyniki transient
- **compare_grad_vs_uni10** — GRAD vs UNI10 w czasie + steady ref.
- **compare_uni10_005_coldfluid** — efekt czynnika zimnego (woda vs olej vs HFE).

### Braki / decyzje
| # | Kwestia | Do zrobienia |
|---|---|---|
| C-a | **Niepewność na wykresach wyników** (pasma GCI na fig_F/bar) | propagacja GCI% → error bars |
| C-b | **Zakres artykułu**: AW-only (steady) czy z AO/HFE (transient)? | decyzja autora |
| C-c | **Rola transientu**: główny wątek czy demonstracja możliwości? | decyzja autora |
| C-d | **Skala mocy** (~7 W/wymiennik) — znormalizować (per komórka / per m³)? | rozważyć q_vol (fig_I już jest) |

### Do artykułu (rekomendacja figur głównych)
1. fig_A (profile — z akcentem na Re(z) GRAD)
2. fig_C (Δp)
3. fig_B2 (wskaźnik niezależny od Dh) — zamiast lub obok fig_B
4. fig_F (bar summary) LUB fig_E (radar) — jedno z nich
5. fig_D (mapa efektywności)
Reszta (H, I, L, E/F) → supplementary.

---

## D. Tabele i liczby do zestawienia

1. **Warunki brzegowe / punkt pracy** (❗ do uzupełnienia z case Fluent):
   - ṁ_air, ṁ_water [kg/s] — **BRAK, wyekstrahować** (journal ich nie zawiera).
   - T_in: air ≈ 298 K, water = 363 K (mamy z CSV).
   - Właściwości fluidów (ρ, μ, k, cp, Pr) — częściowo w SRP.
   - Re: GRAD air 341 / water 314; UNI10 air 218 / water 221 (mamy).
2. **Geometria**: rozmiary komórek, porowatość ε (≈0.42), Dh (GRAD ≈8.7 mm, UNI10 ≈5.4 mm), pole wymiany.
3. **Zbieżność siatki**: N, h, Nu, f + GCI% per geometria.
4. **Bilans energii**: Q_air, Q_cold, imbalance% (mamy: ±0.05%).
5. **Kluczowe metryki GRAD vs UNI10** (z FIGURES_DESCRIPTION — h, Nu, f, Nu/f^⅓, Q/Ppump).
6. **Efekt czynnika zimnego**: h_F2 woda/olej/HFE (mamy z transientu: ~3700 / ~1100 / ~1200).

---

## E. Otwarte decyzje (dla autora)

1. **Czasopismo docelowe** (determinuje format V&V i liczbę figur).
2. **Zakres**: tylko Air/Water steady, czy z AO/HFE i transientem?
3. **Walidacja**: pozostajemy przy korelacjach, czy jest szansa na eksperyment z Brna?
4. **Hot/cold**: potwierdzić kierunek (woda gorąca / powietrze zimne wg BC steady) i ujednolicić opisy.
5. **Model turbulencji w narracji**: laminar jako bazowy (Re~200–350), EARSM jako sprawdzenie wrażliwości?

---

## F. Rekomendowana kolejność prac (co odblokowuje co)

1. **Eksport z Fluent** (must-have, blokuje sekcję Verification): residua, y+, właściwości/BC (ṁ).
2. **Domknąć Verification z danych, które mamy**: figura+tabela bilansu energii (gotowe liczby),
   GCI dla Fluid2 (A1), tabela zbieżności (A3), studium kroku czasowego z part3 (B).
3. **Ustalić zakres i czasopismo** (E1–E2) — determinuje selekcję figur.
4. **Finalizacja figur wyników**: wybór main vs supplementary, spójny styl, error bars z GCI (C-a).
5. **Napisać sekcje** w kolejności: Model numeryczny → Grid Independence → V&V → Wyniki.

### Must-have dla recenzji CFD (bez tego odrzut/major revision)
- [x] Bilans energii (mamy: ±0.05%)
- [x] Grid independence + GCI (mamy dla F1/UNI10; uzupełnić F2 + narracja GRAD)
- [ ] Residua / iterative convergence — **eksport z Fluent**
- [ ] y+ dla EARSM — **eksport z Fluent**
- [ ] Time-step independence (transient) — z istniejących danych
- [x] Walidacja vs korelacje (mamy; jawnie zaznaczyć brak eksperymentu jako ograniczenie)
- [ ] Tabela warunków brzegowych z ṁ — **eksport z Fluent**
