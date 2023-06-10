using SimIntensiveInference

include("problem_setup.jl")
include("plotting.jl")

n = 1000

logps, us = run_es(f, g, π, L, n)

# Calculate the mean and standard deviation of the (log-)permeabilities
μ_post = reshape(mean(logps[:,:,end], dims=2), g_c.nx, g_c.ny)
σ_post = reshape(std(logps[:,:,end], dims=2), g_c.nx, g_c.ny)

fname = "plots/darcy_flow/es/es_mean.pdf"
plot_μ_post(μ_post, logps_t, g_c, g_f, fname)

fname = "plots/darcy_flow/es/es_stds.pdf"
plot_σ_post(σ_post, g_c, fname)