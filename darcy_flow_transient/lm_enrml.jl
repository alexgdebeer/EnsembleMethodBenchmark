using SimIntensiveInference

include("steadystate_channel.jl")

Ne = 100
γ = 10
i_max = 8

θs, us, Ss, λs = run_lm_enrml(f, g, p, L, γ, i_max, Ne)

# logps = θs[1:end,:,:]
# μ_post = reshape(mean(logps[:,:,end], dims=2), grid.nx, grid.ny)

logps = reduce(hcat, get_perms(p, θ) for θ ∈ eachcol(θs[:,:,end]))
μ_post = reshape(mean(logps, dims=2), grid.nx, grid.ny)