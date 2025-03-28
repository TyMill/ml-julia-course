# 🧠 Machine Learning with Julia – `ml-julia-course`

[![DOI](https://zenodo.org/badge/955290469.svg)](https://doi.org/10.5281/zenodo.15100006)
![Version](https://img.shields.io/github/v/release/TyMill/ml-julia-course)
![License](https://img.shields.io/github/license/TyMill/ml-julia-course)
![Language](https://img.shields.io/github/languages/top/TyMill/ml-julia-course)




📘 **ml-julia-course** to praktyczny kurs uczenia maszynowego w języku Julia, zawierający komplet 17 notatników prowadzących krok po kroku od podstaw języka do zaawansowanego AutoML.

---

## 📂 Zawartość kursu

| Nr | Temat | Plik |
|----|-------|------|
| 01 | Wprowadzenie do języka Julia | `01_intro_to_julia.jl` |
| 02 | Praca z DataFrames i CSV | `02_dataframes_csv.jl` |
| 03 | Wizualizacja danych | `03_visualization.jl` |
| 04 | Regresja liniowa | `04_linear_regression.jl` |
| 04a | Wprowadzenie do MLJ | `04_mlj_basics.jl` |
| 05 | Regresja logistyczna | `05_logistic_regression.jl` |
| 06 | Drzewa decyzyjne | `06_decision_tree_classifier.jl` |
| 07 | Random Forest | `07_random_forest_classifier.jl` |
| 08 | K-Nearest Neighbors | `08_knn_classifier.jl` |
| 09 | Naiwny klasyfikator Bayesa | `09_naive_bayes_classifier.jl` |
| 10 | XGBoost | `10_xgboost_classifier.jl` |
| 11 | Porównanie modeli | `11_model_comparison.jl` |
| 12 | PCA i redukcja wymiarowości | `12_pca_dimensionality_reduction.jl` |
| 13 | Klasteryzacja KMeans | `13_kmeans_clustering.jl` |
| 14 | Przypadek użycia: SynthPred | `14_synthpred_usecase.jl` |
| 15 | Interpretacja modeli | `15_model_interpretation.jl` |
| 16 | Podsumowanie kursu | `16_notebooks_summary.jl` |
| 17 | AutoML z MLJ | `17_automl_with_mlj.jl` |

---

## 🚀 Jak uruchomić

1. Zainstaluj Julię: [https://julialang.org/downloads](https://julialang.org/downloads)
2. W konsoli Julia:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

3. Uruchom notatnik:

- Jupyter: `jupyter notebook`
- Pluto:  
```julia
using Pluto
Pluto.run()
```

---

## 🎯 Cele kursu

- Poznanie języka Julia w kontekście analizy danych
- Zbudowanie praktycznych modeli ML (regresja, klasyfikacja, klasteryzacja)
- Nauka redukcji wymiarowości i interpretacji modeli
- Porównanie modeli i automatyzacja uczenia maszynowego

---

## 🔗 Powiązane projekty

- [SynthPred.jl](https://github.com/TyMill/SynthPred) – narzędzie do predykcji z użyciem danych syntetycznych

---

## 📚 Dla kogo?

- Studenci kierunków technicznych
- Analitycy danych chcący poznać Julię
- Nauczyciele i dydaktycy poszukujący materiałów dydaktycznych
- Każdy, kto chce zrozumieć ML przez praktykę

---

## 📜 Licencja

MIT

---

**Autor**: Tymoteusz Miller  
📍 Uniwersytet Szczeciński
📫 [LinkedIn](https://www.linkedin.com/in/tymoteuszmiller) | [Zenodo Profile](https://zenodo.org/search?page=1&size=20&q=TyMill)
