include("setup.jl")
include("InferenceAlgorithms/eki.jl")

Ne = 1000
ηs, θs, Fs, Gs, αs = run_eki_dmc(F, G, pr, d_obs, μ_e, Γ_e, L_e, Ne)

μ_post = reshape(mean(θs[end], dims=2), grid_c.nx, grid_c.nx)
σ_post = reshape(std(θs[end], dims=2), grid_c.nx, grid_c.nx)

# Hyperparameters
σs_pri = [pr.σ_bounds[1] + cdf(Normal(), σ) * pr.Δσ for σ ∈ ηs[1][end-1, :]]
ls_pri = [pr.l_bounds[1] + cdf(Normal(), l) * pr.Δl for l ∈ ηs[1][end, :]]
σs_post = [pr.σ_bounds[1] + cdf(Normal(), σ) * pr.Δσ for σ ∈ ηs[end][end-1, :]]
ls_post = [pr.l_bounds[1] + cdf(Normal(), l) * pr.Δl for l ∈ ηs[end][end, :]]