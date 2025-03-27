# 02_dataframes_csv.jl
# Praca z danymi tabelarycznymi – CSV i DataFrames
# Autor: Tymoteusz Miller

#######################################################
# WYMAGANE PAKIETY
#######################################################

using CSV
using DataFrames

#######################################################
# WCZYTYWANIE DANYCH Z CSV
#######################################################

# Jeśli nie masz pliku CSV, możesz utworzyć tymczasowy DataFrame i go zapisać
sample_df = DataFrame(Name = ["Anna", "Jan", "Kasia"], Age = [23, 34, 29], Score = [88.5, 92.0, 79.5])
CSV.write("sample_data.csv", sample_df)

# Wczytywanie pliku CSV
df = CSV.read("sample_data.csv", DataFrame)
println("Zawartość wczytanego DataFrame:")
println(df)

#######################################################
# PODSTAWOWE OPERACJE NA DATAFRAME
#######################################################

# Wyświetlanie kolumn
println("Kolumny: ", names(df))

# Podstawowe statystyki
describe(df)

# Filtrowanie danych
df_filtered = filter(row -> row.Score > 85, df)
println("Filtr: Score > 85")
println(df_filtered)

# Sortowanie
df_sorted = sort(df, :Age, rev=true)
println("Sortowanie po wieku malejąco:")
println(df_sorted)

# Dodawanie kolumny
df.Total = df.Score .* 1.1
println("Dodano kolumnę Total (Score * 1.1):")
println(df)

#######################################################
# ZAPIS DO NOWEGO PLIKU CSV
#######################################################

CSV.write("processed_data.csv", df)
println("Zapisano przetworzony DataFrame do 'processed_data.csv'")
