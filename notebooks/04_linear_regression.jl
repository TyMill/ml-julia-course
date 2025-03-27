# 04_linear_regression.jl
# Regresja liniowa z MLJ.jl – predykcja zmiennej ciągłej
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
# PRZYGOTOWANIE DANYCH – PRZYKŁAD UCZELNIA/EDUKACJA
# Predykcja średniej oceny na podstawie czasu nauki
#######################################################

study_hours = [1.0, 2.0, 3.0, 4.5, 5.0, 6.5, 7.0, 8.0, 9.5, 10.0]
final_score = [52, 55, 60, 65, 67, 75, 78, 80, 88, 90]

df = DataFrame(Hours=study_hours, Score=final_score)

# Rozdzielenie danych
X = select(df, Not(:Score))
y = df.Score

#######################################################
# BUDOWA MODELU REGRESJI LINIOWEJ
#######################################################

LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
model = LinearRegressor()

# Maszyna i trenowanie
mach = machine(model, X, y)
fit!(mach)

# Parametry modelu
fitted_params = fitted_params(mach)
println("Współczynniki modelu:")
println(fitted_params)

#######################################################
# PREDYKCJA I WIZUALIZACJA
#######################################################

y_pred = predict(mach, X)

# Przekształcenie predykcji do wektora Float64
y_pred_float = MLJ.matrix(y_pred)[:, 1]

# Wykres
scatter(X.Hours, y, label="Rzeczywiste dane", xlabel="Godziny nauki", ylabel="Ocena końcowa")
plot!(X.Hours, y_pred_float, label="Predykcja regresji liniowej", lw=2)

#######################################################
# EWALUACJA MODELU
#######################################################

# Metryki: R^2, RMSE
r2_val = r_squared(y_pred, y)
rmse_val = rms(y_pred, y)

println("\nEwaluacja modelu regresji:")
println("R² = ", round(r2_val, digits=4))
println("RMSE = ", round(rmse_val, digits=4))

# Przykład predykcji na nowym rekordzie
new_X = DataFrame(Hours=[6.0, 8.5])
new_y_pred = predict(mach, new_X)
println("\nPredykcja dla 6.0 i 8.5h nauki:")
println(new_y_pred)
