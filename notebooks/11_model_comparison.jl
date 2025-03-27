# 11_model_comparison.jl
# Porównanie modeli uczenia maszynowego w MLJ.jl
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I IMPORT PAKIETÓW
#######################################################

# using Pkg
# Pkg.add("MLJ")
# Pkg.add("DataFrames")
# Pkg.add("ROCAnalysis")
# Pkg.add("Plots")

using MLJ
using MLJLinearModels
using DecisionTree
using XGBoost
using ROCAnalysis
using DataFrames
using Plots

#######################################################
# DANE: PRZYKŁAD CHURN (jak w XGBoost)
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
# DEFINICJA MODELI
#######################################################

models = [
    (@load LogisticClassifier pkg=MLJLinearModels verbosity=0)(),
    (@load DecisionTreeClassifier pkg=DecisionTree verbosity=0)(max_depth=4),
    (@load XGBoostClassifier pkg=XGBoost verbosity=0)(nrounds=50, eta=0.1, max_depth=4)
]

model_names = ["Logistic", "DecisionTree", "XGBoost"]
results = []

#######################################################
# TRENING, PREDYKCJA, EWALUACJA
#######################################################

for (i, model) in enumerate(models)
    mach = machine(model, X, y)
    fit!(mach)
    yhat = predict(mach, X)
    yhat_labels = predict_mode(mach, X)
    acc = accuracy(yhat_labels, y)
    auc = auc(roc(yhat, y))
    push!(results, (Model=model_names[i], Accuracy=acc, AUC=auc))
end

#######################################################
# TABELA WYNIKÓW
#######################################################

results_df = DataFrame(results)
println("Porównanie modeli:")
println(results_df)

#######################################################
# WYKRES ROC
#######################################################

plot()
for (i, model) in enumerate(models)
    mach = machine(model, X, y)
    fit!(mach)
    yhat = predict(mach, X)
    r = roc(yhat, y)
    plot!(r, label=model_names[i])
end

xlabel!("False Positive Rate")
ylabel!("True Positive Rate")
title!("Krzywe ROC – porównanie modeli")
