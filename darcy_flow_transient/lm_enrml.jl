using SimIntensiveInference

include("setup_ss_channel.jl")

αs = [16.0 for _ ∈ 1:16]
n = 100

γ = 10
i_max = 16

θs, us, Ss, λs = run_lm_enrml(f, g, p, L, γ, i_max, n)

logps = θs[6:end,:,:]
μ_post = reshape(mean(logps[:,:,end], dims=2), grid.nx, grid.ny)