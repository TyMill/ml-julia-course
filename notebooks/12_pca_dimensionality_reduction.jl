# 12_pca_dimensionality_reduction.jl
# Redukcja wymiarowości z PCA – Principal Component Analysis
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I IMPORT PAKIETÓW
#######################################################

# using Pkg
# Pkg.add("MLJ")
# Pkg.add("MultivariateStats")
# Pkg.add("DataFrames")
# Pkg.add("Plots")

using MLJ
using MultivariateStats
using DataFrames
using Plots

#######################################################
# DANE PRZYKŁADOWE – CECHY SYNTETYCZNE
#######################################################

df = DataFrame(
    Feature1 = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
    Feature2 = [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9],
    Feature3 = [1.0, 2.1, 1.3, 3.3, 2.2, 1.0, 2.8, 1.4, 2.0, 1.6]
)

X = coerce(df, autotype(df, :discrete_to_continuous))

#######################################################
# PCA – REDUKCJA DO 2 WYMIARÓW
#######################################################

PCA = @load PCA pkg=MultivariateStats
model = PCA(maxoutdim=2)

mach = machine(model, X)
fit!(mach)

X_transformed = transform(mach, X)
df_pca = DataFrame(X_transformed, :auto)
rename!(df_pca, [:PC1, :PC2])

println("Pierwsze dwie składowe główne (PCA):")
println(df_pca)

#######################################################
# WIZUALIZACJA W 2D
#######################################################

scatter(df_pca.PC1, df_pca.PC2,
    xlabel="PC1", ylabel="PC2",
    title="Redukcja wymiarowości PCA (2D)",
    legend=false
)
