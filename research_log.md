# Research Log

Ten plik sluzy do sledzenia postepu projektu i decyzji technicznych.

Kazdy istotny wpis powinien zawierac:
- date i godzine lokalna,
- krotki opis celu,
- liste zmienionych plikow lub wygenerowanych wynikow,
- najwazniejsze decyzje i zalozenia,
- sposob weryfikacji,
- otwarte pytania lub rzeczy do sprawdzenia pozniej.

Zalecany format wpisu:

```markdown
## YYYY-MM-DD HH:MM TZ - krotki tytul

Cel:
- ...

Zmiany:
- ...

Wyniki:
- ...

Weryfikacja:
- ...

Uwagi / nastepne kroki:
- ...
```

---

## 2026-05-07 22:34 +02:00 - Wykresy zbieznosci G5, G6 i G7

Cel:
- Rozbudowac analize zbieznosci siatki dla geometrii GRAD i UNI10.
- Pokazac procentowe odchylenia wybranych metryk od najgestszej siatki.
- Przygotowac wykresy w formie bardziej czytelnej do dalszego review, w tym do sprawdzenia przez Claude LLM.

Zmiany:
- Dodano helper `_pct_dev(...)` w `Steady_Repo/mesh_convergence_plots.py`.
- Dodano wykresy G5:
  - `fig_G5_grad_dt_dp_dev.png` - osobny wykres GRAD, 4 panele: `dT_F1`, `dT_F2`, `dp_F1`, `dp_F2`.
  - `fig_G5_uni10_dt_dp_dev.png` - osobny wykres UNI10, 4 panele: `dT_F1`, `dT_F2`, `dp_F1`, `dp_F2`.
- Dodano wykres G6:
  - `fig_G6_combined_dt_dp_dev.png` - dwa panele, lewy GRAD i prawy UNI10; szara linia dla `dT`, niebieska dla `dp`; kolka dla Fluid1, kwadraty dla Fluid2.
- Oczyszczono G6:
  - usunieto tekst `References...` pod wykresem,
  - usunieto tytul legendy `G6 encoding`,
  - usunieto z legendy pozycje `Selected mesh`,
  - usunieto z legendy pozycje o `GRAD M005/M006/M007`,
  - usunieto podpisy/nazwy siatek z paneli.
- Dodano wykres G7:
  - `fig_G7_combined_h_f_dev.png` - analogiczny do G6, ale dla `h` i `f`.
- Zaktualizowano opis wykresow w `Steady_Repo/DataProcessed/plots/FIGURES_DESCRIPTION.md`.

Wyniki:
- Wygenerowane pliki:
  - `Steady_Repo/DataProcessed/plots/fig_G5_grad_dt_dp_dev.png`
  - `Steady_Repo/DataProcessed/plots/fig_G5_uni10_dt_dp_dev.png`
  - `Steady_Repo/DataProcessed/plots/fig_G6_combined_dt_dp_dev.png`
  - `Steady_Repo/DataProcessed/plots/fig_G7_combined_h_f_dev.png`

Zalozenia:
- Dla GRAD wartosc referencyjna to najgestsza siatka `M004`.
- Dla UNI10 wartosc referencyjna to najgestsza siatka `GUNI_008`.
- Odchylenie liczone jest jako `(value_i - value_ref) / value_ref * 100%`.
- W G6 i G7 punkty `M005/M006/M007` dla GRAD pozostaja jako dodatkowe markery model-study przy 881K, ale bez podpisow i bez osobnej pozycji w legendzie.

Weryfikacja:
- Uruchomiono `python mesh_convergence_plots.py` w `Steady_Repo`.
- Skrypt zakonczyl sie poprawnie i wygenerowal G5, G6 oraz G7.
- Sprawdzono, ze wygenerowane PNG sa niepuste.
- Sprawdzono numerycznie, ze punkty referencyjne:
  - `M004` dla GRAD,
  - `GUNI_008` dla UNI10,
  maja `0.0%` dla odpowiednich metryk G5, G6 i G7.

Uwagi / nastepne kroki:
- Claude LLM ma sprawdzic czy forma G6/G7 jest wystarczajaco czytelna i czy dodatkowe markery `M005/M006/M007` nie zaciemniaja interpretacji.
- Przy kolejnych zmianach w repo nalezy dopisywac nowy wpis do tego pliku z data, godzina i opisem wykonanych prac.

---

## 2026-05-08 14:30 +02:00 - Dodanie punktow transient do G6 i G7

Cel:
- Dodac wyniki transient do wykresow G6 i G7 jako gwiazdki bez laczenia linia.
- Wyciagnac wartosci usrednione z istniejacych wynikow transient zamiast wpisywac je recznie.

Zmiany:
- Rozszerzono `Steady_Repo/mesh_convergence_plots.py` o odczyt transient:
  - `h`, `f` i `dp_sum` sa liczone jako srednia z ostatnich 20 wierszy plikow `summary_means__*.csv`.
  - `dT` jest liczone z ostatnich 20 plikow SRP jako roznica temperatury masowej miedzy pierwsza i ostatnia plaszczyzna.
- Dodano gwiazdki transient do:
  - `fig_G6_combined_dt_dp_dev.png`,
  - `fig_G7_combined_h_f_dev.png`.

Zrodla danych:
- GRAD AW Laminar: `Transient_Repo/TransientFigs/csv/job_grad_part3/` oraz `Transient_Repo/FluentTransientData/part3/`.
- GRAD AW k-omega/EARSM: `Transient_Repo/TransientFigs/csv/job_grad_part4/` oraz `Transient_Repo/FluentTransientData/part4/`.
- UNI10 003/005/007: odpowiednie katalogi `job_uni10_*` oraz `FluentTransientData/uni10_*`.

Weryfikacja:
- Uruchomiono `python mesh_convergence_plots.py` w `Steady_Repo`.
- Skrypt wygenerowal G6 i G7 bez bledow.
- Sprawdzono, ze nowe PNG sa niepuste.
- Sprawdzono, ze wartosci transient dla `dT`, `dp`, `h` i `f` sa policzone dla wszystkich przypadkow bez `NaN`.

Uwagi / nastepne kroki:
- Claude LLM powinien sprawdzic, czy gwiazdki transient nie sa mylone z punktami steady i czy legenda jest wystarczajaco czytelna.

---

## 2026-05-08 14:34 +02:00 - Jawne odfiltrowanie transientow air/oil

Cel:
- Upewnic sie, ze wykresy G6/G7 nie uwzgledniaja transientow GRAD wykonanych dla pary air/oil.
- Pozostawic tylko przypadki air/water zgodne z porownaniem steady.

Zmiany:
- W `Steady_Repo/mesh_convergence_plots.py` dodano pole `fluid_pair="air_water"` do definicji transientow uzywanych jako gwiazdki.
- Dodano jawny filtr w `_build_transient_star_values()`, ktory pomija kazdy transient bez `fluid_pair == "air_water"`.

Wyniki:
- W G6/G7 pozostaja tylko:
  - `AW_Lam` z `part3`,
  - `AW_kw` z `part4`,
  - `uni10_003`,
  - `uni10_005`,
  - `uni10_007`.
- Przypadki GRAD air/oil (`part5`-`part9`) nie sa uwzgledniane.

Weryfikacja:
- Uruchomiono `python mesh_convergence_plots.py` w `Steady_Repo`.
- Skrypt wygenerowal G6 i G7 bez bledow.
- Wypisano liste transientow uzytych przez skrypt i potwierdzono, ze wszystkie maja `fluid_pair=air_water`.

---

## 2026-05-08 14:50 +02:00 - Poprawa definicji transient dp w G6

Cel:
- Wyjasnic, dlaczego punkty GRAD transient pokazywaly odchylenia `dp` rzedu 35-45%.
- Poprawic obliczanie `dp` dla transient, aby bylo zgodne z definicja steady.

Problem:
- Dla transient `dp` bylo pobierane z kolumny `mean_...p_sum[Pa]` w plikach `summary_means`.
- Ta wartosc jest srednia po pasmach z kolumny skumulowanego spadku cisnienia, a nie calkowitym spadkiem inlet-outlet.
- Dlatego wartosc transient `dp` wychodzila sztucznie zanizona, mniej wiecej do polowy wartosci steady.

Zmiany:
- W `Steady_Repo/mesh_convergence_plots.py` transient `dp` jest teraz liczony z ostatnich 20 plikow SRP jako `abs(P_in - P_out)`, analogicznie do steady.
- `dT` pozostaje liczone z SRP jako `abs(T_out - T_in)`.
- `h` i `f` pozostaja liczone z `summary_means` jako srednia z ostatnich 20 krokow.

Wyniki po poprawce:
- GRAD AW Laminar `dp_F1`: ok. `+0.99%` wzgledem M004.
- GRAD AW Laminar `dp_F2`: ok. `-0.81%` wzgledem M004.
- GRAD AW k-omega `dp_F1`: ok. `-0.04%` wzgledem M004.
- GRAD AW k-omega `dp_F2`: ok. `-1.09%` wzgledem M004.

Weryfikacja:
- Uruchomiono `python mesh_convergence_plots.py` w `Steady_Repo`.
- G6 i G7 wygenerowaly sie bez bledow.
- PNG sa niepuste.

---

## 2026-07-07 +02:00 - Faza 0: synchronizacja dokumentacji z kodem i danymi (reprodukowalnosc)

Cel:
- Usunac dryf miedzy plikami `.md` a `config.py`/danymi na dysku, aby dokumentacja == kod == dane.
- Fundament pod dalsze porownania (Fazy 1-3). Cel nadrzedny: porzadek i reprodukowalnosc.

Zmiany (tylko dokumentacja, brak zmian w kodzie/danych):
- Root `CLAUDE.md`:
  - Tabela "Fluid Pairs Simulated" rozszerzona: Air/Oil (transient), dodana para **Air/HFE**;
    dopisano kolumne "Where" i note, ze steady = tylko Air/Water.
  - "Naming Conventions": dodano `part10`, `uni10_005_oil`, `uni10_005_hfe`.
- `Transient_Repo/CLAUDE.md`:
  - Tabela CASES uzupelniona o `grad_aw_laminar_part10`, `uni10_005_hfe` (Air/HFE),
    `uni10_005_oil` (Air/Oil), z zakresami czasu i `t_interpretation`.
  - Tabela "Active Jobs" zsynchronizowana z `config.py` (m.in. `compare_grad_uni10_001` = aktywny,
    `job_grad_part1..9` = nieaktywne, `job_grad_part10` = aktywny); dodano note, ze istnienie
    katalogu output nie oznacza aktywnosci joba.
  - Sekcja "Running": nota o rozjezdzie sciezek `DEFAULT_*` (`My Drive\...`) vs katalog roboczy
    (`Documents\PK and PR\...`) i zalecenie `--in-dir/--out-dir`.
- `Steady_Repo/DataProcessed/plots/FIGURES_DESCRIPTION.md`:
  - Dopisano sekcje "Uzupelnienie" z opisami wczesniej nieudokumentowanych figur:
    B2 (h/dp^1/3), G1_v2/G2_v2/G3_v2 (dT,dp), G4_v1/v2/v3 (dp/dT/Q_total), H, I, J, K, L.
- Nowy plik `Transient_Repo/FIGURES_DESCRIPTION.md`:
  - Opis struktury `TransientFigs/` (plots/Compare/convergence/csv), konwencji nazw,
    znaczenia linii `*_SS` (steady) oraz tabela aktualnych COMPARE_JOBS.

Wyniki:
- Zmienione: root `CLAUDE.md`, `Transient_Repo/CLAUDE.md`, `Steady_Repo/.../FIGURES_DESCRIPTION.md`.
- Utworzone: `Transient_Repo/FIGURES_DESCRIPTION.md`.

Weryfikacja:
- Tabele CASES/Active Jobs porownane 1:1 z `config.py` (`CASES`, `PLOT_JOBS`, `COMPARE_JOBS`).
- Kazda figura obecna w `Steady_Repo/DataProcessed/plots/` ma teraz wpis w FIGURES_DESCRIPTION.md.
- Nazwy plikow w opisie transient potwierdzone z rzeczywistych katalogow `plots/`, `Compare/`, `csv/`.

Uwagi / nastepne kroki:
- Faza 1: aktywacja i generacja `compare_grad_vs_uni10` (obecnie `active=False`).
- Faza 2: nowy COMPARE_JOB "efekt czynnika zimnego" (UNI10 005 woda vs olej vs HFE).
- Faza 3: dedykowana figura walidacji transient -> steady (reuse `_build_transient_star_values`).
- Otwarte pytanie do wlasciciela repo: ktora lokalizacja repo jest kanoniczna (My Drive vs Documents).

---

## 2026-07-07 +02:00 - Faza 1: aktywacja i generacja compare_grad_vs_uni10 (+ fix bug "SS SS")

Cel:
- Domknac najsilniejsze merytorycznie porownanie: GRAD vs UNI10 (transient, air/water) z referencjami steady.

Problem wykryty przy pierwszym uruchomieniu:
- `compare_grad_vs_uni10` mial etykiety steady z jawnym sufiksem " SS" (np. "M006 SS"),
  a `compare_runner.py` (overlay) automatycznie dokleja " SS" -> "M006 SS SS" -> KeyError w `style.py`.
- Dzialajace joby (compare_grad_uni10_001) uzywaja etykiet steady BEZ " SS", z kluczami style_map z " SS".

Zmiany:
- `config.py`: `compare_grad_vs_uni10` -> `active=True`.
- `config.py`: w bloku `steady.cases` tego joba usunieto sufiks " SS" z etykiet
  ("M006 SS"->"M006", "GUNI 003 SS"->"GUNI 003", itd.); klucze `style_map` ("M006 SS", ...) bez zmian.
  Dodano komentarz wyjasniajacy konwencje (runner dokleja " SS").
- `Transient_Repo/CLAUDE.md` i `Transient_Repo/FIGURES_DESCRIPTION.md`: oznaczenie joba jako aktywny (✓).

Uruchomienie:
- `python main.py --in-dir ./FluentTransientData --out-dir ./TransientFigs --jobs compare_grad_vs_uni10`
  (jawne sciezki lokalne, bo `DEFAULT_*` wskazuja na My Drive).

Wyniki:
- 8 figur w `TransientFigs/Compare/compare_grad_vs_uni10/`:
  `cmp_overlay_{h,f}_{F1,F2}` + `cmp_mean_{h,f}_{F1,F2}__aligned`. Wszystkie niepuste (0.28-1.67 MB).

Weryfikacja:
- Skrypt zakonczyl sie bez bledow (8x [OK]).
- Podglad `cmp_mean_h_F1`: GRAD (part1-4) ~55 W/m2K pokrywa sie z M006/M007 SS;
  UNI10 (003/005/007) ~60 pokrywa sie z liniami GUNI SS. Spojne ze steady.

Uwagi / nastepne kroki:
- UWAGA interpretacyjna: GRAD i UNI10 pracuja w roznych punktach Re (F1 ~341 vs ~218) - patrz Steady fig_F.
  Bezposrednie porownanie h/f nalezy czytac lacznie z Re. Zaznaczone w FIGURES_DESCRIPTION.md.
- Job pozostawiony `active=True` (reprodukowalnosc: `python main.py` odtwarza figury).

---

## 2026-07-07 +02:00 - Faza 2: porownanie efektu czynnika zimnego (UNI10 005: woda/olej/HFE)

Cel:
- Wykorzystac przetworzone dane oleju i HFE do izolowanego porownania wplywu czynnika strony zimnej
  na h i f, przy identycznej geometrii/siatce/biegu (UNI10 005).

Zmiany:
- `config.py`: nowy COMPARE_JOB `compare_uni10_005_coldfluid` (active=True).
  Serie: `uni10_005` (Water), `uni10_005_oil` (Oil), `uni10_005_hfe` (HFE), oba fluidy (F1, F2).
  Okna czasu z `UNI_COMPARE_WINDOWS_S`. Steady wylaczony celowo (tylko woda ma GUNI_005 - asymetria).
- `Transient_Repo/CLAUDE.md` + `FIGURES_DESCRIPTION.md`: dodany wpis o jobie i interpretacja.

Uruchomienie:
- `python main.py --in-dir ./FluentTransientData --out-dir ./TransientFigs --jobs compare_uni10_005_coldfluid`

Wyniki:
- 8 figur w `TransientFigs/Compare/compare_uni10_005_coldfluid/` (overlay + mean, h/f, F1/F2). Niepuste.

Weryfikacja:
- 8x [OK], bez bledow.
- Podglad `cmp_mean_h_F2`: woda ~3700 W/m2K, olej i HFE ~1100-1200 W/m2K - fizycznie poprawne
  (wyzsza przewodnosc/Pr wody => wyzsze h). Wyrazna separacja czynnikow.

Uwagi / nastepne kroki:
- Faza 3: dedykowana figura walidacji transient -> steady.

---

## 2026-07-07 +02:00 - Faza 3: dedykowana figura walidacji transient -> steady (G8)

Cel:
- Zebrac rozproszona walidacje (gwiazdki w G6/G7) w jedna czytelna figure + artefakt liczbowy CSV.

Decyzja projektowa:
- Figura powstala w `Steady_Repo/mesh_convergence_plots.py`, bo tam sa slowniki referencji steady
  (GRAD, UNI, DTD_P) oraz logika tail-mean transient (`TRANSIENT_STAR_VALUES`). Budowa w Transient_Repo
  oznaczalaby duplikacje. G6/G7 pozostaja nietkniete (nowa, niezalezna funkcja).

Zmiany:
- `mesh_convergence_plots.py`: dodano `TRANSIENT_TO_STEADY`, `G8_CASE_LABELS`, `_steady_metric_g8()`
  oraz `fig_G8_transient_vs_steady()` (+ wywolanie w __main__).
  Mapowanie: AW_Lam->M006, AW_kw->M007, uni10_003/005/007->GUNI_003/005/007.
  Metryki: h, f, dT, dp (F1 i F2). Reuzycie `TRANSIENT_STAR_VALUES` i `_pct_dev`.
- `Steady_Repo/.../FIGURES_DESCRIPTION.md`: dodany opis G8.

Wyniki:
- `fig_G8_transient_vs_steady.png` (4 panele slupkowe % odchylenia, pasmo +/-2%).
- `fig_G8_transient_vs_steady.csv` (case, steady_mesh, metric, fluid, transient, steady, pct_dev).

Weryfikacja:
- `python mesh_convergence_plots.py` - wszystkie figury G1..G8 [OK], bez bledow.
- CSV sprawdzony numerycznie:
  - h, dT: wszystkie przypadki w +/-2% wzgledem steady (silna zgodnosc transient<->steady).
  - f_F1 i dp_F1 dla AW_kw (EARSM): -5.0% / -3.3% - poza pasmem, zgodne ze znana ~5.3% wrazliwoscia
    tarcia modelu EARSM (Steady CLAUDE.md). Pozostale f/dp w +/-2.5%.

Uwagi / nastepne kroki:
- Wszystkie 4 fazy planu (0-3) zrealizowane. Otwarte pytanie do wlasciciela: kanoniczna lokalizacja
  repo (My Drive vs Documents) - do potwierdzenia przy ujednoliceniu `DEFAULT_*` w Transient config.py.

---

## 2026-07-07 +02:00 - Plan przygotowania do artykulu + kontrola bilansu energii

Cel:
- Wykonac prompt "zaplanuj przygotowanie wynikow do artykulu" (Grid Independence / V&V / Wyniki).
- Przy okazji policzyc najwazniejszy brak weryfikacyjny: bilans energii Q_air vs Q_cold.

Zmiany:
- Nowy plik `ARTICLE_PREP_PLAN.md` (mapa zasobow -> sekcje, rejestr brakow, plan figur, tabele, decyzje).

Kontrola bilansu energii (sum |Q_band|, z DataProcessed/csv):
- GRAD M006:      Q_air=7.041 W,  Q_cold=7.045 W,  imbalance=-0.05%
- GRAD M007:      Q_air=7.052 W,  Q_cold=7.051 W,  imbalance=+0.03%
- UNI10 GUNI_007: Q_air=7.639 W,  Q_cold=7.643 W,  imbalance=-0.05%
=> Bilans domyka sie ~0.05% (mocny wynik weryfikacyjny).

Ustalenia fizyczne:
- dT_F2 ~ 0.08 K NIE jest bledem: Q~7 W, woda ma ogromne m*cp -> minimalny przyrost T. Spojne z bilansem.
- Punkt pracy steady: T_in air ~298 K, T_in water=363 K. Woda wchodzi gorąca i grzeje powietrze.
- UWAGA: to przeczy opisowi "air hot side" w CLAUDE.md - do rozstrzygniecia/ujednolicenia.
- Re (mean): GRAD air 341 / water 314; UNI10 air 218 / water 221 -> rezim laminarny/przejsciowy.

Braki potwierdzone (do eksportu z Fluent, must-have dla recenzji CFD):
- residua / iterative convergence, y+ (istotne dla EARSM), warunki brzegowe z m_dot,
  systematyczne studium kroku czasowego (dane dt=1e-4/2.5e-4/5e-4 istnieja).

Weryfikacja:
- Liczby bilansu policzone z Sim_Data_Fluid{1,2}_{M006,M007,GUNI_007}.csv (kolumny Q_band, Q_total).
- Naglowek CSV potwierdzony (Q_band[W], Q_total[W], T_bulk_band[K], Re[-]).

---

## 2026-07-07 +02:00 - GCI dla obu fluidow (UNI10) + ocena figur zbieznosci

Cel:
- Rozszerzyc GCI o Fluid2 (dotad tylko Fluid1) i ocenic mozliwosc GCI dla GRAD.

Zmiany:
- `mesh_convergence.py`: `plot_uni10_convergence` liczy teraz GCI takze dla Fluid2 (para 007/008).
  Nowe klucze w `gci_results.csv`: h_F2/Nu_F2/f_F2 _2mesh.

Wyniki (UNI10, 2-mesh 007/008, p=2, Fs=1.25):
- h_F1=1.06% Nu_F1=1.08% f_F1=2.27% | h_F2=0.78% Nu_F2=0.86% f_F2=0.38%.
- Wszystkie <=2.27% -> siatka GUNI_007 (6.25M) zbiezna dla obu fluidow.

GRAD - brak obronialnego GCI:
- Seria M001-M004 confounded (BL + N razem). Nawet M006(881K) vs M004(12.5M) daje h_F2 -5.8%,
  ale M004 to k-omega a M006 laminar -> odchylenie miesza siatke z modelem. GCI dla GRAD niewiarygodne.
- Adekwatnosc siatki GRAD opiera sie na: (1) niewrazliwosci na model turb. przy 881K (M005/006/007),
  (2) analogii do UNI10 (drobniejsze struktury UNI10 wymagaja wiecej komorek -> 881K wystarcza dla GRAD).

Ocena figur do publikacji:
- conv_uni10/conv_grad: F1(air~60) i F2(water~3700) na jednej osi -> krzywa powietrza splaszczona. NIE do papieru.
- G3/G5/G6/G7 (% odchylenia): czytelne, ale zatloczone gwiazdkami transient + model-study. Do papieru
  potrzebna WYCZYSZCZONA wersja (tylko seria siatek + pasmo +-2% + linia zalecanej siatki).
- Rekomendacja: 1 figura zbieznosci = % odchylenie, seria siatek only, F1/F2 jako ksztalt markera.

---

## 2026-07-08 +02:00 - Tabele GCI do artykulu (Article_text/data) + generator

Cel:
- Wygenerowac table-ready CSV do sekcji Grid Independence: siatki Coarse/Medium/Fine,
  parametry dT, h, dp, f (Fluid1+Fluid2), blad wzgledny WZGLEDEM Medium, + transient dla Medium.

Zmiany:
- Nowy folder `Article_text/data/` + generator `Article_text/make_gci_tables.py`.
  Reuzywa mcp.GRAD/UNI (h,f), mcp.DTD_P (dT,dp), mcp.TRANSIENT_STAR_VALUES (tail-mean transient).
  Triplety siatek konfigurowalne w TRIPLETS (dobrane z ISTNIEJACYCH danych).
- Wyjscie: `data/gci_uni10.csv`, `data/gci_grad.csv`.

Triplety (z istniejacych danych; docelowe 8M/6M nie istnieja -> kampania):
- UNI10: Coarse GUNI_005 (2.6M) / Medium GUNI_007 (6.25M) / Fine GUNI_008 (16.7M). Transient dla Medium = uni10_007.
- GRAD: Coarse M002_v2 (906K) / Medium M002_5 (3.0M) / Fine M003_v2 (8.4M), wszystkie laminar. Transient brak (tylko 881K).

Wyniki i interpretacja:
- UNI10 - historia dziala: eps(Fine vs Medium) h_F1=-0.79%, h_F2=+0.58% (male);
  eps(Coarse vs Medium) h_F1=+2.38%, h_F2=+1.74% (wieksze) -> Medium (6.25M) wystarcza.
  Transient vs Medium: h_F1 +0.84%, dT_F1 +0.09% -> steady potwierdzony w transiencie.
- GRAD - historia NIE dziala (potwierdza brak zbieznosci): eps(Fine vs Medium) h_F2=-9.3%,
  h_F1=-3.4%; Fine odbiega od Medium bardziej niz Coarse. Dwie siatki ~900K laminar: 55.3 vs 57.0.
  => GRAD wymaga nowej czystej serii laminarnej (~2/6/14M) + transientow. Tabela oznaczona PROVISIONAL.

Weryfikacja:
- Uruchomiono `python make_gci_tables.py`; oba CSV zapisane, liczby zgodne z reczna kontrola GCI.

Uwagi / nastepne kroki:
- Docelowe Medium 8M (UNI) / 6M (GRAD) + transienty = kampania Fluent (do zaplanowania).
- f_F1 dla UNI zaszumione (fine ~ coarse) - typowe dla tarcia.

---

## 2026-07-10 +02:00 - Aktualizacja G3 v2-v5: M003_v4 transient part11 + M008 steady

Cel:
- Dopiac aktualny zestaw wykresow zbieznosci GRAD/UNI po dodaniu steady `M008` i dlugiego
  transientu `part11` dla `Mesh_003_v4`.

Zmiany w danych i mapowaniu:
- `part11` opisany jako `grad_aw_laminar_part11`, GRAD Air/Water Laminar, `Mesh_003_v4`,
  7,895,738 komorek, dt=5e-4, zakres 0.02-20.0 s; okno porownawcze 18-19 s.
- `part_desc.txt` uzupelniony o steady referencje `M003_v4_SS` i `M008_SS`.
- `M003_v4 transient` na wykresach pochodzi z `Transient_Repo/TransientFigs/csv/job_grad_part11/`.
- `M008` steady dodany do serii GRAD laminar: 6,018,034 komorek.

Wartosci M008 (z `Sim_Data_Fluid1_M008.srp` / `Sim_Data_Fluid2_M008.srp`):
- Fluid1: h=57.192591 W/m2K, Nu=19.203538, f=1.117518, dp_sum=8.169012 Pa.
- Fluid2: h=3442.612935 W/m2K, Nu=45.182371, f=1.028158, dp_sum=8.195765 Pa.

Figury:
- `fig_G3_combined_conv_v2.png`: h/f, ograniczona liczba elementow, legenda pod wykresem.
- `fig_G3_combined_conv_v3.png`: dT/dp dla tego samego zestawu porownawczego.
- `fig_G3_combined_conv_v4.png`: h/f, GRAD tylko laminar, UNI bez zmian.
- `fig_G3_combined_conv_v5.png`: h/f w wartosciach bezwzglednych.

Interpretacja robocza:
- `part10` zostaje jako krotki historyczny eksport `Mesh_003_v4`, ale dla aktualnych wykresow
  zbieznosci preferowany jest dlugi `part11`.
- `M003_v4`/`part11` jest teraz para steady/transient dla tej samej rodziny siatki.
- `M008` wypelnia serie GRAD ok. 6M komorek i pomaga ocenic przebieg pomiedzy `M002.5`
  a najgestszymi punktami.

Weryfikacja:
- `python -m py_compile Steady_Repo\mesh_convergence_plots.py` - OK.
- Regeneracja `fig_G3_combined_conv_v2/v3/v4/v5` z `mesh_convergence_plots.py` - OK.
- `python Transient_Repo\main.py --in-dir Transient_Repo\FluentTransientData --out-dir Transient_Repo\TransientFigs --jobs job_grad_part11` - OK.

---

## 2026-07-24 10:19 +02:00 - G3 v6: tortuosity i czas przebywania

Cel:
- Przygotowac panel porownujacy rozklady tortuosity, czas przebywania, lokalne kwantyle
  `tau50/IQR/tau90` oraz udzial trajektorii trapped dla GRAD i UNI10.

Zmiany:
- Dodano `Steady_Repo/tortuosity_v6.py`.
- Dodano kontrakt danych i pusty szablon:
  - `Steady_Repo/DataProcessed/tortuosity/README.md`,
  - `Steady_Repo/DataProcessed/tortuosity/trajectories_TEMPLATE.csv`.
- Dodano `fig_G3_combined_conv_v6()` do `mesh_convergence_plots.py`.
- Rozszerzono `Steady_Repo/DataProcessed/plots/FIGURES_DESCRIPTION.md`.

Decyzje:
- Tortuosity jest liczone jako `L_path/L_straight >= 1`; odwrotnosc jest raportowana jako
  straightness.
- Rozklady sa liczone wylacznie z zakonczonych trajektorii, a `N_trapped/N_in` ze wszystkich
  unikalnych czastek.
- Brak trajektorii w aktualnym repo nie jest uzupelniany proxy z SRP. Generator tworzy
  oznaczona figure `WAITING FOR DATA`, dopoki nie pojawi sie `trajectories.csv`.

Weryfikacja:
- `python -m py_compile tortuosity_v6.py mesh_convergence_plots.py` - OK.
- Test kontrolny w pamieci: `N_in`, `N_completed`, `N_trapped`, `tau50` i `tau90` - OK.
- Pelny test panelu na sztucznym zbiorze 4 grup po 60 trajektorii - OK; sprawdzono PNG.
- Uruchomienie bez `trajectories.csv` - OK; wygenerowano jawny stan `WAITING FOR DATA`.

Uwagi / nastepne kroki:
- Wyeksportowac z Fluent pathlines/particle tracks dla obu domen i kierunkow przeplywu,
  lacznie z identyfikatorem czastki, segmentem, dlugoscia drogi i czasem przebywania.

---

## 2026-09-02 +02:00 - Panele na spotkanie z partnerami (Results/Plot/Partners_panel)

Cel:
- Jeden zestaw figur po angielsku na spotkanie z partnerami: wersja pelna 2x3 i skrocona 2x2,
  bez watku kretosci (tortuosity) - wylacznie dane pasmowe (Run 001) + transient (Run 002).

Zmiany:
- Nowy folder `Results/Plot/Partners_panel/` z `_data.py` (warstwa danych + fity + outliery)
  i `make_partner_panels.py` (generator obu paneli).
- Zrodla danych: gotowe CSV z `Run001_two_simu` (band_values, summary R000-R005)
  oraz `Run002_transient_fluids` (transient_global_summary, pec, energy_balance).
  Fizyka nie jest przeliczana ponownie.

Panele:
- (a) Nu(Re) log-log, obie strony, punkty globalne + fit + R2, w tle chmura pasmowa.
- (b) dp(U) log-log z wykladnikami i linia odniesienia Darcy n=1.
- (c) test rozprzezenia stron (odchylenie od sredniej klastra) + replikat R000/R005.
- (d) znormalizowany profil h/h_sr vs x/L, 12 przebiegow + srednie kierunkowe.
- (e) h_powietrza vs h_cieczy dla 3 cieczy x 2 geometrie, tabelka Q / P_pump.
- (f) PEC (h,dp) i (Nu,f), GRAD/UNI10, + ramka walidacyjna.
- 2x2 = (a), (d), (e), (f).

Decyzje:
- R2 pokazywane WYLACZNIE dla korelacji globalnych; jakosc dopasowania na poziomie pasm
  celowo nie jest raportowana na figurach.
- Outliery odseparowane jednym waskim kryterium: h < 10% mediany run/strona (kolaps LMTD).
  Odrzucone: 1 pasmo (R003 / Water, band 20, h = 13.9 W/m2K, h/mediana = 0.009).
  Prog jest o rzad wielkosci nizszy niz najslabsze fizyczne pasmo (0.39 mediany), wiec
  realny spadek h na koncu rdzenia nie jest usuwany. Lista w `outliers_excluded.csv`.

Wyniki liczbowe (zapisane w `correlations_global.csv` i `panel_key_numbers.csv`):
- Powietrze: Nu = 0.461 Re^0.642 (R2 = 0.9981, Re 341-1596); dp = 22.84 U^1.875 (R2 = 0.9995).
- Woda:      Nu = 0.690 Re^0.724 (R2 = 0.9982, Re 100-315);  dp = 2112 U^1.727 (R2 = 0.9999).
- Rozprzezenie: h_air spread 2.34% (woda 3.2x), h_cold spread 5.76% (powietrze 4.7x),
  replikat R000 vs R005 = 0.022%.
- Profile pasmowe: powietrze -29..-41%, rozrzut 2.2-2.8x; woda -24..-53%, rozrzut 2.4-5.7x.
- Ciecze zimne: h zmienia sie 3.7x, Q = 7.02-7.65 W; UNI10 olej/woda = 3.9x mocy pompowania.
- Walidacja: bilans energii 0.33-0.67% (srednia na przypadek), <= 1.51% najgorszy krok;
  fluktuacja czasowa h: powietrze 1.2-1.7%, ciecz 0.02-1.14%.

Weryfikacja:
- `python make_partner_panels.py` - OK; PNG + PDF dla obu paneli, 3 CSV.
- Obejrzano oba PNG; poprawiono kolizje etykiet w (c), (d), (e).

Uwagi / nastepne kroki:
- Panel bez kretosci ustawia scene pod kolejny krok (brakujaca zmienna lokalna);
  slabsze dopasowanie na poziomie pasm zostaje jako material na pytania, nie na figure.

## 2026-09-03  Wariant panelu 2x2 z definicja Nu (make_partner_panels_nudef.py)

Nowy skrypt Results/Plot/Partners_panel/make_partner_panels_nudef.py -> panel_partners_2x2_nudef.{png,pdf}
+ panel_key_numbers_nudef.csv. Reuzywa _data.py i paneli (b)/(c)/(d) z make_partner_panels.py;
nadpisuje tylko panel (a).

Panel (a) dostal:
- definicje Nu = h * D_h / k_fluid w ramce (gora-prawo);
- inset "Water / Air ratio (band medians)": h ×21.4 w gore, D_h/k ÷25 w dol,
  iloczyn Nu ×0.85 ~ parytet -> pokazuje wprost, ze podobne Nu przy roznym Re
  bierze sie z wzajemnego znoszenia sie duzego h wody i malego D_h/k wody.
Liczby z band_values_R000..R005: mediany pasm h_water/h_air=20.7, (D_h/k)_water/(D_h/k)_air=0.040,
Nu_water/Nu_air=0.83. D_h/k odzyskane jako Nu/h (nic nie przeliczane).

Zakresy D_h (przeliczone z tablic pasmowych SRP, compute_band_table, Dh[m]):
- powietrze (Fluid1): 4.66-14.80 mm po pasmach, mediana 7.95 mm
- woda (Fluid2): 7.06-9.11 mm po pasmach, mediana 7.83 mm
D_h praktycznie identyczne miedzy cieczami (mediany 7.95 vs 7.83, ratio 0.98) i stale
miedzy runami (geometria, nie przeplyw) -> caly stosunek D_h/k (÷25) to k, nie D_h.
Ramka definicji w panelu (a) mowi to wprost.
Stary panel_partners_2x2.* nietkniety.
