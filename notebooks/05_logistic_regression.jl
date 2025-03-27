# 05_logistic_regression.jl
# Regresja logistyczna z MLJ.jl – klasyfikacja binarna (np. zdanie egzaminu)
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I IMPORT PAKIETÓW
#######################################################

# using Pkg
# Pkg.add("MLJ")
# Pkg.add("MLJLinearModels")
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("Plots")

using MLJ
using MLJLinearModels
using DataFrames
using CSV
using Plots

#######################################################
# DANE PRZYKŁADOWE – STUDIA/ZDANIE EGZAMINU
# Predykcja: czy student zda egzamin (1) lub nie (0)
#######################################################

study_hours = [1.0, 2.0, 3.0, 4.5, 5.0, 6.0, 7.5, 8.0, 9.0, 10.0]
passed_exam = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

df = DataFrame(Hours=study_hours, Passed=passed_exam)

X = select(df, Not(:Passed))
y = coerce(df.Passed, OrderedFactor)

#######################################################
# BUDOWA MODELU REGRESJI LOGISTYCZNEJ
#######################################################

LogisticRegressor = @load LogisticClassifier pkg=MLJLinearModels
model = LogisticRegressor()

mach = machine(model, X, y)
fit!(mach)

#######################################################
# PREDYKCJA I WIZUALIZACJA
#######################################################

y_pred = predict(mach, X)  # probabilistyczna predykcja

# Konwersja do prawdopodobieństw i etykiet
prob = MLJ.matrix(y_pred)[:, 2]
pred_labels = predict_mode(mach, X)

# Wykres
scatter(X.Hours, df.Passed, label="Rzeczywiste", xlabel="Godziny nauki", ylabel="Zdał (1) / Nie (0)", ylim=(-0.1,1.1))
plot!(X.Hours, prob, label="Prawdopodobieństwo zdania", lw=2)

#######################################################
# EWALUACJA MODELU
#######################################################

accuracy_val = accuracy(pred_labels, y)
println("Accuracy: ", round(accuracy_val, digits=4))

# Przykład predykcji na nowych danych
new_X = DataFrame(Hours=[3.0, 6.5, 8.5])
new_pred = predict(mach, new_X)
new_labels = predict_mode(mach, new_X)

println("\nNowe przypadki:")
println("Prawdopodobieństwa: ", new_pred)
println("Klasy: ", new_labels)
