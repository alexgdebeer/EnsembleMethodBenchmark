using SimIntensiveInference

include("setup.jl")

Ne = 100
γ = 10
i_max = 8

θs, us, Ss, λs = run_lm_enrml(f, g, p, L, γ, i_max, Ne)

μ_post = reshape(mean(θs[:,:,end], dims=2), grid_c.nx, grid_c.ny)
σ_post = reshape( std(θs[:,:,end], dims=2), grid_c.nx, grid_c.ny)