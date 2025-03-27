# 15_model_interpretation.jl
# Interpretacja modeli ML – Feature Importance i predykcje
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
# PRZYKŁADOWE DANE – KLIENT/RYZYKO
#######################################################

df = DataFrame(
    Age = [25, 30, 35, 40, 45, 50, 55, 60],
    Income = [2000, 2500, 3000, 3200, 4000, 4100, 4200, 4300],
    CreditScore = [500, 520, 550, 580, 600, 620, 640, 660],
    Risk = ["High", "High", "Medium", "Medium", "Low", "Low", "Low", "Low"]
)

X = select(df, Not(:Risk))
y = coerce(df.Risk, Multiclass)

#######################################################
# MODEL: RandomForestClassifier
#######################################################

RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
model = RandomForestClassifier(n_trees=100)

mach = machine(model, X, y)
fit!(mach)

#######################################################
# INTERPRETACJA – WAŻNOŚĆ CECH
#######################################################

fp = fitted_params(mach)
feature_importance = fp.feature_importances

println("Ważność cech (feature importance):")
for (name, score) in zip(names(X), feature_importance)
    println(name, ": ", round(score, digits=4))
end

#######################################################
# WYKRES WAŻNOŚCI CECH
#######################################################

bar(names(X), feature_importance,
    xlabel="Cecha", ylabel="Waga",
    title="Ważność cech w klasyfikatorze Random Forest")

#######################################################
# PREDYKCJA DLA NOWEGO KLIENTA
#######################################################

new_client = DataFrame(Age=[38], Income=[3300], CreditScore=[590])
pred = predict_mode(mach, new_client)

println("\nPredykcja ryzyka dla nowego klienta:")
println(pred)
