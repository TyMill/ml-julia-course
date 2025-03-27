# 10_xgboost_classifier.jl
# Klasyfikator XGBoost z MLJ.jl – predykcja churnu klienta
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I IMPORT PAKIETÓW
#######################################################

# using Pkg
# Pkg.add("MLJ")
# Pkg.add("XGBoost")
# Pkg.add("DataFrames")
# Pkg.add("Plots")

using MLJ
using XGBoost
using DataFrames
using Plots

#######################################################
# PRZYKŁAD: CHURN – CZY KLIENT ZREZYGNUJE Z USŁUGI?
# Dane: wiek, liczba miesięcy aktywności, liczba reklamacji
# Cel: Churn (tak / nie)
#######################################################

df = DataFrame(
    Age = [22, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    MonthsActive = [12, 24, 36, 8, 5, 15, 6, 9, 3, 1],
    Complaints = [0, 1, 0, 3, 5, 1, 4, 2, 6, 7],
    Churned = ["No", "No", "No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes"]
)

X = select(df, Not(:Churned))
y = coerce(df.Churned, Multiclass)

#######################################################
# BUDOWA MODELU XGBOOST
#######################################################

XGBClassifier = @load XGBoostClassifier pkg=XGBoost
model = XGBClassifier(nrounds=50, eta=0.1, max_depth=4)

mach = machine(model, X, y)
fit!(mach)

#######################################################
# PREDYKCJA I EWALUACJA
#######################################################

y_pred = predict_mode(mach, X)
acc = accuracy(y_pred, y)

println("Predykcje churn:")
println(y_pred)
println("Accuracy: ", round(acc, digits=4))

#######################################################
# PREDYKCJA DLA NOWYCH KLIENTÓW
#######################################################

new_clients = DataFrame(
    Age = [32, 48, 62],
    MonthsActive = [18, 4, 2],
    Complaints = [1, 3, 6]
)

new_pred = predict_mode(mach, new_clients)
println("\nNowi klienci:")
println(new_pred)

#######################################################
# UWAGA: XGBoost to model o wysokiej wydajności – warto eksperymentować z parametrami
# np. eta, max_depth, booster, nrounds itd.
