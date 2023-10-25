include("setup.jl")
include("InferenceAlgorithms/eki.jl")

NF = grid_c.nx^2 * grid_c.nt
Ne = 1000

ηs, Fs, Gs, αs = run_eki_dmc(F_r, G, pr, y_obs, μ_ε, Γ_e, NF, Ne)

lnps = hcat([transform(pr, η) for η in eachcol(ηs[end])]...)

μ_post = reshape(mean(lnps, dims=2), grid_c.nx, grid_c.nx)
σ_post = reshape(std(lnps, dims=2), grid_c.nx, grid_c.nx)

# Hyperparameters
σs_pri = [pr.σ_bounds[1] + cdf(Normal(), σ) * pr.Δσ for σ ∈ ηs[1][end-1, :]]
ls_pri = [pr.l_bounds[1] + cdf(Normal(), l) * pr.Δl for l ∈ ηs[1][end, :]]
σs_post = [pr.σ_bounds[1] + cdf(Normal(), σ) * pr.Δσ for σ ∈ ηs[end][end-1, :]]
ls_post = [pr.l_bounds[1] + cdf(Normal(), l) * pr.Δl for l ∈ ηs[end][end, :]]