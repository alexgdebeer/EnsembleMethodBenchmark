using SimIntensiveInference

include("problem_setup.jl")

# Form perturbation kernel
Γ_K = 0.02^2 * Γ_π
K = MvNormal(Γ_K)

# Define the number of chains to run and the number of samples to generate 
N = 1_000_000
n_chains

logps, us = @time SimIntensiveInference.run_mcmc(
    f, g, π, L, K, N, n_chains=n_chains, θ_s=vec(ps_true)
)

@save "data/darcy_flow/coarse_grid/mcmc_results.jld2" logps us