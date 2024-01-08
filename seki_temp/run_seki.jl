include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

Ne = 100
   
θs = run_seki(F, G, pr, d_obs, μ_e, Γ_e, Ne)

μ_post = transform(pr, mean(θs, dims=2))
μ_post = reshape(μ_post, grid_c.nx, grid_c.nx)

# σ_post = std(us[end], dims=2)
# σ_post = reshape(σ_post, grid_c.nx, grid_c.nx)