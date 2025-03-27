# 17_automl_with_mlj.jl
# AutoML z MLJ.jl – automatyczne strojenie i selekcja modelu
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I IMPORT PAKIETÓW
#######################################################

# using Pkg
# Pkg.add("MLJ")
# Pkg.add("MLJTuning")
# Pkg.add("DataFrames")
# Pkg.add("XGBoost")
# Pkg.add("DecisionTree")
# Pkg.add("Plots")
# Pkg.add("Statistics")

using MLJ
using MLJTuning
using DataFrames
using XGBoost
using DecisionTree
using Statistics
using Plots

#######################################################
# PRZYKŁAD: KLASA CZY NIE? (CHURN)
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
# DEFINICJA MODELU: XGBoost
#######################################################

XGBClassifier = @load XGBoostClassifier pkg=XGBoost
base_model = XGBClassifier()

# Zakres hiperparametrów
range_eta = range(base_model, :eta, lower=0.01, upper=0.3)
range_max_depth = range(base_model, :max_depth, lower=2, upper=8)

#######################################################
# DEFINICJA STRATEGII AUTO ML
#######################################################

tuned_model = TunedModel(
    model=base_model,
    resampling=CV(nfolds=5, shuffle=true),
    tuning=RandomSearch(),
    range=[range_eta, range_max_depth],
    measure=accuracy,
    n=25, # liczba iteracji
    acceleration=CPUThreads()
)

mach = machine(tuned_model, X, y)
fit!(mach)

#######################################################
# WYNIKI STROJENIA
#######################################################

best_model = fitted_params(mach).best_model
println("🎯 Najlepszy model XGBoost:")
println(best_model)

# Ewaluacja na zbiorze treningowym
y_pred = predict_mode(mach, X)
acc = accuracy(y_pred, y)
println("Dokładność najlepszego modelu: ", round(acc, digits=4))

#######################################################
# ZAPIS I PREDYKCJA
#######################################################

new_client = DataFrame(
    Age = [38],
    MonthsActive = [12],
    Complaints = [2]
)

new_pred = predict_mode(mach, new_client)
println("\nPredykcja dla nowego klienta:")
println(new_pred)

#######################################################
# DODATKOWO: KRZYWA ROC DLA MODELU Z TUNINGU
#######################################################

y_prob = predict(mach, X)
roc_curve = roc(y_prob, y)

plot(roc_curve,
    title="AutoML: Krzywa ROC – XGBoost z RandomSearch",
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    legend=false
)

#######################################################
# WSKAZÓWKI I NAJLEPSZE PRAKTYKI
#######################################################

println("\n📌 Wskazówki AutoML:")
println("- Możesz użyć GridSearch lub RandomSearch")
println("- Użyj resamplingu (CV, Holdout) dla lepszej oceny generalizacji")
println("- Połączenie AutoML z interpretacją modeli daje potężne narzędzie")

println("\n✅ Gotowe – zautomatyzowana selekcja i tuning zakończone sukcesem.")
