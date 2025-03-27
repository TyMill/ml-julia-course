# 01_intro_to_julia.jl
# Wprowadzenie do języka Julia
# Autor: Tymoteusz Miller

#######################################################
# PODSTAWY SKŁADNI I OBLICZEŃ
#######################################################

# Komentarze w Julia zaczynają się od #
println("Witaj w języku Julia!")

# Proste zmienne
a = 5
b = 2.5
c = a + b
println("a + b = ", c)

# Typy danych
typeof(a)       # Int64
typeof(b)       # Float64
typeof("tekst") # String

# Operacje matematyczne
x = 10
y = 3
println("Dzielenie: ", x / y)
println("Dzielenie całkowite: ", div(x, y))
println("Reszta z dzielenia: ", mod(x, y))

#######################################################
# FUNKCJE
#######################################################

# Prosta funkcja
function pow2(x)
    return x^2
end

println("4^2 = ", pow2(4))

# Funkcja anonimowa
cube = x -> x^3
println("3^3 = ", cube(3))

#######################################################
# PRACA Z PAKIETAMI
#######################################################

# Instalacja pakietu (jeśli nie masz go jeszcze zainstalowanego, odkomentuj poniższe)
# using Pkg
# Pkg.add("DataFrames")

# Ładowanie pakietu
using DataFrames

# Tworzenie prostego DataFrame
df = DataFrame(A = 1:3, B = ["a", "b", "c"])
println(df)

