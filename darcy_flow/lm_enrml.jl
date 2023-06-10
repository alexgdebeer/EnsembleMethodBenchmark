using SimIntensiveInference

include("problem_setup.jl")

γ = 10
i_max = 16
n = 1000

logps, us, Ss, λs = run_lm_enrml(f, g, π, L, γ, i_max, n)

# Calculate the mean and standard deviation of the (log-)permeabilities
μ_post = reshape(mean(logps[:,:,end], dims=2), g_c.nx, g_c.ny)
σ_post = reshape(std(logps[:,:,end], dims=2), g_c.nx, g_c.ny)

fname = "plots/darcy_flow/enrml/gn_enrml_mean.pdf"
plot_μ_post(μ_post, logps_t, g_c, g_f, fname)

fname = "plots/darcy_flow/enrml/gn_enrml_stds.pdf"
plot_σ_post(σ_post, g_c, fname)