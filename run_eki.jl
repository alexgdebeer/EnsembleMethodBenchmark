include("setup.jl")
include("InferenceAlgorithms/eki.jl")

NF = grid_c.nx^2 * grid_c.nt
Ne = 100

ηs, Fs, Gs, αs = run_eki_dmc(F_r, G, pr, y_obs, μ_ε, Γ_e, NF, Ne)

lnps = hcat([transform(pr, η) for η in eachcol(ηs[end])]...)

μ_post = reshape(mean(lnps, dims=2), grid_c.nx, grid_c.nx)
σ_post = reshape(std(lnps, dims=2), grid_c.nx, grid_c.nx)