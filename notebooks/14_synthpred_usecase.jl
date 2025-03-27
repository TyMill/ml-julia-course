# 14_synthpred_usecase.jl
# Generowanie danych syntetycznych z SynthPred.jl + ML pipeline
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I IMPORT PAKIETÓW
#######################################################

# using Pkg
# Pkg.add("SynthPred")
# Pkg.add("MLJ")
# Pkg.add("DataFrames")
# Pkg.add("Plots")

using SynthPred
using MLJ
using DataFrames
using Plots

#######################################################
# GENEROWANIE DANYCH SYNTHETYCZNYCH
#######################################################

# Wygeneruj dane dla klasyfikacji binarnej
df, meta = generate_synthetic_data(
    n_samples=100,
    n_features=5,
    problem_type="classification",
    class_balance=0.6,
    noise_level=0.1,
    random_state=42
)

println("Pierwsze 5 wierszy danych:")
first(df, 5)

# Wyodrębnienie X i y
X = select(df, Not(:target))
y = coerce(df.target, Multiclass)

#######################################################
# MODELOWANIE – RandomForestClassifier
#######################################################

RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
model = RandomForestClassifier(n_trees=100)

mach = machine(model, X, y)
fit!(mach)

y_pred = predict_mode(mach, X)
acc = accuracy(y_pred, y)

println("\nDokładność modelu na danych syntetycznych: ", round(acc, digits=4))

#######################################################
# WIZUALIZACJA DLA 2 PIERWSZYCH CECH
#######################################################

scatter(X[:, 1], X[:, 2], group=y,
    xlabel="Feature 1", ylabel="Feature 2",
    title="Dane syntetyczne z SynthPred – klasy",
    legend=false
)
