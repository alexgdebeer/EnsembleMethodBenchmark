using SimIntensiveInference

include("problem_setup.jl")
include("plotting.jl")

# Define inflation factors
αs = [16.0 for _ ∈ 1:16]
N_e = 1000

logps, us = SimIntensiveInference.run_es_mda(f, g, π, L, αs, N_e)

# Calculate the mean and standard deviation of the (log-)permeabilities
μ_post = reshape(mean(logps[:,:,end], dims=2), g_c.nx, g_c.ny)
σ_post = reshape(std(logps[:,:,end], dims=2), g_c.nx, g_c.ny)

fname = "plots/darcy_flow/es/mda_mean.pdf"
plot_μ_post(μ_post, logps_t, g_c, g_f, fname)

fname = "plots/darcy_flow/es/mda_stds.pdf"
plot_σ_post(σ_post, g_c, fname)