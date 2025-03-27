# 06_decision_tree_classifier.jl
# Klasyfikator drzewa decyzyjnego z MLJ.jl
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I IMPORT PAKIETÓW
#######################################################

# using Pkg
# Pkg.add("MLJ")
# Pkg.add("DecisionTree")
# Pkg.add("DataFrames")
# Pkg.add("Plots")

using MLJ
using DecisionTree
using DataFrames
using Plots

#######################################################
# PRZYKŁAD: AKTYWNOŚĆ UŻYTKOWNIKÓW NA PODSTAWIE WIEKU I CZASU ONLINE
#######################################################

df = DataFrame(
    Age = [18, 22, 25, 30, 34, 40, 45, 50],
    OnlineTime = [0.5, 1.0, 2.0, 2.5, 1.5, 0.7, 0.3, 0.1],
    Active = ["No", "Yes", "Yes", "Yes", "Yes", "No", "No", "No"]
)

X = select(df, Not(:Active))
y = coerce(df.Active, Multiclass)

#######################################################
# BUDOWA MODELU DRZEWA DECYZYJNEGO
#######################################################

TreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
model = TreeClassifier(max_depth=3)

mach = machine(model, X, y)
fit!(mach)

#######################################################
# PREDYKCJA I EWALUACJA
#######################################################

y_pred = predict_mode(mach, X)
acc = accuracy(y_pred, y)

println("Predykcje:")
println(y_pred)
println("Dokładność klasyfikacji: ", round(acc, digits=4))

#######################################################
# PREDYKCJA DLA NOWYCH UŻYTKOWNIKÓW
#######################################################

new_users = DataFrame(Age=[28, 36, 20], OnlineTime=[2.1, 0.4, 1.5])
new_pred = predict_mode(mach, new_users)
println("\nNowi użytkownicy:")
println(new_pred)

#######################################################
# (OPCJONALNIE) WIZUALIZACJA WAŻNOŚCI CECH
#######################################################

fp = fitted_params(mach)
println("\nWażność cech (feature importance):")
println(fp.feature_importances)
