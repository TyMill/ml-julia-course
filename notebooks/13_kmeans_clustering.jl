# 13_kmeans_clustering.jl
# Grupowanie nienadzorowane z KMeans – MLJ.jl
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I IMPORT PAKIETÓW
#######################################################

# using Pkg
# Pkg.add("MLJ")
# Pkg.add("Clustering")
# Pkg.add("DataFrames")
# Pkg.add("Plots")

using MLJ
using Clustering
using DataFrames
using Plots

#######################################################
# DANE: TE SAME DANE CO W PCA
#######################################################

df = DataFrame(
    Feature1 = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
    Feature2 = [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9],
    Feature3 = [1.0, 2.1, 1.3, 3.3, 2.2, 1.0, 2.8, 1.4, 2.0, 1.6]
)

X = coerce(df, autotype(df, :discrete_to_continuous))

#######################################################
# MODELOWANIE KMEANS (3 KLASTRY)
#######################################################

KMeans = @load KMeans pkg=Clustering
model = KMeans(K=3)

mach = machine(model, X)
fit!(mach)

clusters = predict(mach, X)
cluster_labels = mode.(clusters)

df.Cluster = cluster_labels
println("Przypisane klastry:")
println(df.Cluster)

#######################################################
# WIZUALIZACJA W 2D (Feature1 vs Feature2)
#######################################################

group_color = map(x -> x == 1 ? :red : (x == 2 ? :blue : :green), df.Cluster)

scatter(df.Feature1, df.Feature2, group=group_color,
    xlabel="Feature1", ylabel="Feature2",
    title="Grupowanie KMeans (3 klastry)",
    legend=false
)
