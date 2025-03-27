# 09_naive_bayes_classifier.jl
# Naiwny klasyfikator Bayesa (Naive Bayes) z MLJ.jl
# Autor: Tymoteusz Miller

#######################################################
# INSTALACJA I IMPORT PAKIETÓW
#######################################################

# using Pkg
# Pkg.add("MLJ")
# Pkg.add("NaiveBayes")
# Pkg.add("DataFrames")
# Pkg.add("Plots")

using MLJ
using NaiveBayes
using DataFrames
using Plots

#######################################################
# PRZYKŁAD: PROSTA KLASYFIKACJA TEKSTU (SYNTETYCZNE DANE)
# Cechy: liczba pozytywnych i negatywnych słów w recenzji
# Cel: Klasa (Pozytywna / Negatywna)
#######################################################

df = DataFrame(
    PositiveWords = [3, 2, 5, 4, 0, 1, 0, 0],
    NegativeWords = [0, 1, 0, 1, 4, 3, 5, 6],
    Sentiment = ["Positive", "Positive", "Positive", "Positive", "Negative", "Negative", "Negative", "Negative"]
)

X = select(df, Not(:Sentiment))
y = coerce(df.Sentiment, Multiclass)

#######################################################
# BUDOWA MODELU NAIVE BAYES
#######################################################

NBClassifier = @load GaussianNB pkg=NaiveBayes
model = NBClassifier()

mach = machine(model, X, y)
fit!(mach)

#######################################################
# PREDYKCJA I EWALUACJA
#######################################################

y_pred = predict_mode(mach, X)
acc = accuracy(y_pred, y)

println("Predykcje klas:")
println(y_pred)
println("Dokładność klasyfikacji Naive Bayes: ", round(acc, digits=4))

#######################################################
# PREDYKCJA DLA NOWYCH RECENZJI
#######################################################

new_reviews = DataFrame(
    PositiveWords = [4, 1, 0],
    NegativeWords = [1, 3, 5]
)

new_pred = predict_mode(mach, new_reviews)
println("\nNowe recenzje:")
println(new_pred)

#######################################################
# WIZUALIZACJA – KLASYFIKACJA W 2D
#######################################################

group_color = map(x -> x == "Positive" ? :green : :red, df.Sentiment)
scatter(df.PositiveWords, df.NegativeWords, group=group_color,
    xlabel="Pozytywne słowa",
    ylabel="Negatywne słowa",
    title="Klasyfikacja recenzji: Naive Bayes",
    legend=false
)
