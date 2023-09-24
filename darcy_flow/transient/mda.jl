using SimIntensiveInference

include("setup.jl")

Nf = grid_c.nx * grid_c.ny * (grid_c.nt+1)
Ni = 8
Ne = 500

θs, fs, us, αs, inds = run_es_mda(f, g, p, L, Nf, Ni, Ne, α_method=:constant)

logps = hcat([vec(transform(p, θ)) for θ in eachcol(θs[:,:,end])]...)

μ_post = reshape(mean(logps, dims=2), grid_c.nx, grid_c.ny)
σ_post = reshape(std(logps, dims=2), grid_c.nx, grid_c.ny)