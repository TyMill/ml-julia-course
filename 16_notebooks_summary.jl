# 07_summary.jl - placeholder
# 16_summary_and_resources.jl
# Podsumowanie kursu ML w języku Julia
# Autor: Tymoteusz Miller

#######################################################
# CELE KURSU
#######################################################

# - Wprowadzenie do języka Julia i jego możliwości w ML
# - Poznanie różnych modeli ML: regresja, klasyfikacja, AutoML
# - Analiza danych, redukcja wymiarowości, clustering
# - Interpretacja modeli i wykorzystanie danych syntetycznych
# - Praktyczne zastosowania z użyciem MLJ, SynthPred i innych bibliotek

println("🎓 Kurs ML-Julia zakończony pomyślnie!")

#######################################################
# STRUKTURA KURSU
#######################################################

modules = [
    "01 - Wprowadzenie do Julia",
    "02 - Praca z danymi (CSV, DataFrames)",
    "03 - Wizualizacja danych",
    "04 - Regresja liniowa",
    "05 - Regresja logistyczna",
    "06 - Drzewo decyzyjne",
    "07 - Random Forest",
    "08 - K-Nearest Neighbors",
    "09 - Naiwny Bayes",
    "10 - XGBoost",
    "11 - Porównanie modeli",
    "12 - PCA (redukcja wymiarowości)",
    "13 - KMeans (clustering)",
    "14 - SynthPred i dane syntetyczne",
    "15 - Interpretacja modeli",
    "16 - Podsumowanie i dalsze kroki"
]

println("\n📚 Moduły zawarte w kursie:")
for m in modules
    println("• ", m)
end

#######################################################
# DODATKOWE ZASOBY I LINKI
#######################################################

println("\n🔗 Dalsze źródła i narzędzia:")
println("• Oficjalna dokumentacja MLJ: https://alan-turing-institute.github.io/MLJ.jl/dev/")
println("• Dokumentacja Julia: https://docs.julialang.org/")
println("• Visual Studio Code + Julia Extension")
println("• Pluto.jl (interaktywny notebook) – https://github.com/fonsp/Pluto.jl")
println("• SynthPred.jl (autorska biblioteka): https://github.com/TyMill/SynthPred")
println("• Zenodo DOI: publikacja materiałów dydaktycznych (jeśli dotyczy)")

#######################################################
# KONTAKT I LICENCJA
#######################################################

println("\n✉️ Autor: Tymoteusz Miller – Uniwersytet Szczeciński")
println("📄 Licencja: CC BY-NC-SA 4.0")

println("\nDziękuję za udział w kursie! 🚀")
