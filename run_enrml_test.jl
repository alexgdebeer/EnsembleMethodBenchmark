include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

Ne = 100
    
θs, us, Fs, Gs, Ss, λs, en_ind = run_enrml(
    F, G, pr, d_obs, μ_e, C_e, Ne; 
    localiser=IdentityLocaliser(),
    inflator=AdaptiveInflator()
)

μ_post = transform(pr, mean(θs[end], dims=2))
μ_post = reshape(μ_post, grid_c.nx, grid_c.nx)

σ_post = std(us[end], dims=2)
σ_post = reshape(σ_post, grid_c.nx, grid_c.nx)