# 03_visualization.jl
# Wizualizacja danych w języku Julia – Plots.jl i StatsPlots.jl
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I ŁADOWANIE PAKIETÓW
#######################################################

# using Pkg
# Pkg.add("Plots")
# Pkg.add("StatsPlots")

using Plots
using StatsPlots
using DataFrames

#######################################################
# PRZYGOTOWANIE DANYCH
#######################################################

df = DataFrame(
    Name = ["Anna", "Jan", "Kasia", "Piotr", "Tomek"],
    Age = [23, 34, 29, 41, 38],
    Score = [88.5, 92.0, 79.5, 85.0, 90.2]
)

#######################################################
# WYKRESY PODSTAWOWE
#######################################################

# Wykres punktowy
scatter(df.Age, df.Score,
    xlabel = "Wiek",
    ylabel = "Wynik",
    title = "Zależność: Wiek vs Wynik",
    legend = false
)

# Histogram
histogram(df.Score,
    bins = 5,
    xlabel = "Wynik",
    ylabel = "Liczba osób",
    title = "Rozkład wyników"
)

# Wykres słupkowy
bar(df.Name, df.Score,
    xlabel = "Osoba",
    ylabel = "Wynik",
    title = "Wyniki wg osoby",
    legend = false
)

# Boxplot
@df df boxplot(:Name, :Score,
    ylabel = "Wynik",
    title = "Rozrzut wyników według osoby"
)
