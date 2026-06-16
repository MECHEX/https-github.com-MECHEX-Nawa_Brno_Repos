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
