include("DarcyFlow/DarcyFlow.jl")

using Random
# Random.seed!(0)

xmax, tmax = 100, 1 
Δx, Δt = 2, 0.1

σ = 3.0
l = 30
ν = 1

g = Grid(xmax, tmax, Δx, Δt)
f = MaternFieldKL(g, σ, l, ν, n_modes=2500)

n_samples = 10_000

θs = rand(f, n_samples)
fields = transform(f, θs)

stds = std(fields, dims=2)
heatmap(reshape(stds, g.nx, g.nx), aspect_ratio=:equal, cmap=:turbo)