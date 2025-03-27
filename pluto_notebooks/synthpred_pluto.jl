### A Pluto notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ Cell 1
md"# 🧪 SynthPred.jl – Generowanie danych syntetycznych z interaktywnym eksploratorem"

# ╔═╡ Cell 2
using SynthPred
using DataFrames
using MLJ
using MLJDecisionTreeInterface
using Plots

# ╔═╡ Cell 3
md"## Parametry danych syntetycznych"

# ╔═╡ Cell 4
@bind n_samples Slider(50:25:500, show_value=true, default=200)

# ╔═╡ Cell 5
@bind n_features Slider(2:1:10, show_value=true, default=4)

# ╔═╡ Cell 6
@bind class_balance Slider(0.1:0.1:0.9, show_value=true, default=0.5)

# ╔═╡ Cell 7
@bind noise_level Slider(0.0:0.05:0.5, show_value=true, default=0.1)

# ╔═╡ Cell 8
@bind random_state Slider(1:1:99, show_value=true, default=42)

# ╔═╡ Cell 9
data, meta = generate_synthetic_data(
	n_samples=n_samples,
	n_features=n_features,
	problem_type="classification",
	class_balance=class_balance,
	noise_level=noise_level,
	random_state=random_state
)

# ╔═╡ Cell 10
first(data, 5)

# ╔═╡ Cell 11
X = select(data, Not(:target))
y = coerce(data.target, Multiclass)

# ╔═╡ Cell 12
md"## Trening modelu Random Forest na danych syntetycznych"

# ╔═╡ Cell 13
model = @load RandomForestClassifier pkg=DecisionTree verbosity=0
mach = machine(model, X, y)
fit!(mach)

# ╔═╡ Cell 14
yhat = predict_mode(mach, X)
accuracy(yhat, y)

# ╔═╡ Cell 15
md"## Wizualizacja danych (2 pierwsze cechy)"

# ╔═╡ Cell 16
group_color = map(x -> x == "class_1" ? :blue : :orange, y)
scatter(X[:, 1], X[:, 2],
	group=group_color,
	xlabel="Feature 1",
	ylabel="Feature 2",
	title="Dane syntetyczne (2D)",
	legend=false
)
