using SimIntensiveInference

include("problem_setup.jl")
include("plotting.jl")

n = 100

logps_map, logps = run_rml(f, g, π, L, n)

μ_post = reshape(mean(logps, dims=2), g_c.nx, g_c.ny)
σ_post = reshape(std(logps, dims=2), g_c.nx, g_c.ny)

fname = "plots/darcy_flow/rml/rml_mean.pdf"
plot_μ_post(μ_post, logps_t, g_c, g_f, fname)

fname = "plots/darcy_flow/rml/rml_stds.pdf"
plot_σ_post(σ_post, g_c, fname)
