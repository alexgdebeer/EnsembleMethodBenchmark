using SimIntensiveInference

include("setup.jl")

Ni = 8
Ne = 100

θs, us, αs = run_es_mda(f, g, p, L, Ni, Ne)

μ_post = reshape(mean(θs[:,:,end], dims=2), grid_c.nx, grid_c.ny)
σ_post = reshape(std(θs[:,:,end], dims=2), grid_c.nx, grid_c.ny)