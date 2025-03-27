# 08_knn_classifier.jl
# Klasyfikacja metodą k-NN z MLJ.jl
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I IMPORT PAKIETÓW
#######################################################

# using Pkg
# Pkg.add("MLJ")
# Pkg.add("NearestNeighbors")
# Pkg.add("DataFrames")
# Pkg.add("Plots")

using MLJ
using NearestNeighborModels
using DataFrames
using Plots

#######################################################
# PRZYKŁAD: KLASYFIKACJA KLIENTÓW WG WIEKU I ZAKUPÓW
#######################################################

df = DataFrame(
    Age = [18, 22, 25, 30, 34, 40, 45, 50, 60, 65],
    Purchases = [1, 2, 3, 5, 3, 2, 1, 0, 0, 0],
    Segment = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"]
)

X = select(df, Not(:Segment))
y = coerce(df.Segment, Multiclass)

#######################################################
# BUDOWA MODELU K-NEAREST NEIGHBORS (KNN)
#######################################################

KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
model = KNNClassifier(K=3)

mach = machine(model, X, y)
fit!(mach)

#######################################################
# PREDYKCJA I WIZUALIZACJA
#######################################################

y_pred = predict_mode(mach, X)
acc = accuracy(y_pred, y)

println("Predykcje klas:")
println(y_pred)
println("Dokładność klasyfikacji k-NN: ", round(acc, digits=4))

#######################################################
# PREDYKCJA DLA NOWYCH KLIENTÓW
#######################################################

new_customers = DataFrame(
    Age = [28, 42, 55],
    Purchases = [4, 1, 0]
)

new_pred = predict_mode(mach, new_customers)
println("\nNowi klienci:")
println(new_pred)

#######################################################
# WIZUALIZACJA
#######################################################

group_color = map(x -> x == "A" ? :blue : (x == "B" ? :green : :red), df.Segment)
scatter(df.Age, df.Purchases, group=group_color,
    xlabel="Wiek", ylabel="Liczba zakupów",
    title="Segmentacja klientów metodą k-NN",
    legend=false
)
