# 07_random_forest_classifier.jl
# Random Forest – klasyfikacja z MLJ.jl
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I IMPORT PAKIETÓW
#######################################################

# using Pkg
# Pkg.add("MLJ")
# Pkg.add("DecisionTree")
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("Plots")

using MLJ
using DecisionTree
using DataFrames
using Plots

#######################################################
# PRZYKŁAD: OCENA JAKOŚCI WODY (syntetyczne dane)
# Zmienne: pH, temperatura, azotany -> jakość (dobra/zła)
#######################################################

df = DataFrame(
    pH = [6.5, 7.0, 7.2, 6.8, 8.0, 5.5, 9.1, 6.3, 7.5, 5.9],
    Temperature = [18.0, 19.5, 20.0, 17.0, 22.0, 14.0, 25.0, 16.5, 21.0, 15.0],
    Nitrates = [5.2, 4.8, 3.5, 4.0, 2.5, 8.5, 9.0, 7.8, 2.2, 8.9],
    Quality = ["Good", "Good", "Good", "Good", "Good", "Bad", "Bad", "Bad", "Good", "Bad"]
)

X = select(df, Not(:Quality))
y = coerce(df.Quality, Multiclass)

#######################################################
# BUDOWA MODELU RANDOM FOREST
#######################################################

RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
model = RandomForestClassifier(n_trees=50, max_depth=4)

mach = machine(model, X, y)
fit!(mach)

#######################################################
# PREDYKCJA I EWALUACJA
#######################################################

y_pred = predict_mode(mach, X)
acc = accuracy(y_pred, y)

println("Predykcje:")
println(y_pred)
println("Dokładność modelu Random Forest: ", round(acc, digits=4))

#######################################################
# PREDYKCJA DLA NOWYCH PRÓBEK
#######################################################

new_samples = DataFrame(
    pH = [6.9, 8.2],
    Temperature = [18.5, 23.0],
    Nitrates = [3.0, 9.5]
)

new_pred = predict_mode(mach, new_samples)
println("\nNowe próbki:")
println(new_pred)

#######################################################
# WIZUALIZACJA – WYKRES DECYZYJNYCH KLASYFIKACJI
#######################################################

# Rysunek (2D) – pH vs Nitrates
group_color = map(x -> x == "Good" ? :green : :red, df.Quality)
scatter(df.pH, df.Nitrates, group=group_color,
    xlabel="pH", ylabel="Nitrates",
    title="Jakość wody: Good vs Bad",
    legend=false
)
