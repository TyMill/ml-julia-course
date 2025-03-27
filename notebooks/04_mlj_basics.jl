# 04_mlj_basics.jl
# Uczenie maszynowe z MLJ.jl – podstawy klasyfikacji i regresji
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I WYMAGANE PAKIETY
#######################################################

# using Pkg
# Pkg.add("MLJ")
# Pkg.add("MLJModels")
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("Plots")

using MLJ
using MLJModels
using DataFrames
using CSV
using Plots

#######################################################
# TWORZENIE DANYCH PRZYKŁADOWYCH
#######################################################

# Zbiór danych do regresji (wiek vs wynik testu)
df_reg = DataFrame(
    Age = [18, 22, 25, 30, 35, 40, 45, 50],
    Score = [65, 70, 74, 80, 82, 85, 86, 87]
)

#######################################################
# MODEL REGRESJI
#######################################################

# Definicja cechy i celu
X_reg = select(df_reg, Not(:Score))
y_reg = df_reg.Score

# Wybór modelu
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
model_reg = LinearRegressor()

# Opakowanie modelu
mach_reg = machine(model_reg, X_reg, y_reg)

# Trenowanie modelu
fit!(mach_reg)

# Predykcja
y_pred_reg = predict(mach_reg, X_reg)
println("Predykcja regresji:")
println(y_pred_reg)

# Wizualizacja
scatter(X_reg.Age, y_reg, label="Rzeczywiste", xlabel="Wiek", ylabel="Wynik")
plot!(X_reg.Age, y_pred_reg, label="Predykcja", lw=2, title="Regresja liniowa")

#######################################################
# TWORZENIE DANYCH DO KLASYFIKACJI
#######################################################

df_clf = DataFrame(
    Wiek = [22, 25, 28, 31, 35, 40, 45, 50],
    Aktywny = [1, 1, 0, 0, 1, 0, 0, 1]
)

X_clf = select(df_clf, Not(:Aktywny))
y_clf = df_clf.Aktywny

# Wybór modelu klasyfikacyjnego
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
model_clf = DecisionTreeClassifier(max_depth=3)

# Trening klasyfikatora
mach_clf = machine(model_clf, X_clf, y_clf)
fit!(mach_clf)

# Predykcja
y_pred_clf = predict_mode(mach_clf, X_clf)
println("Predykcja klasyfikacji (tryb):")
println(y_pred_clf)

#######################################################
# EWALUACJA MODELI
#######################################################

# Regressja: błąd średniokwadratowy
rms_val = rms(y_pred_reg, y_reg)
println("RMS (regresja): ", rms_val)

# Klasyfikacja: dokładność
acc_val = accuracy(y_pred_clf, y_clf)
println("Accuracy (klasyfikacja): ", acc_val)

# Macierz błędów
cm = confusion_matrix(y_clf, y_pred_clf)
println("Macierz błędów:")
println(cm)
