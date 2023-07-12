include("setup.jl")

Ne = 25
γ = 10
i_max = 10

θs, fs, us, ss, λs, inds = run_lm_enrml(f, g, p, L, γ, i_max, n_blocks, Ne)

logps_post = reduce(hcat, get_perms(p, θ) for θ ∈ eachcol(θs[:,inds,end]))

μ_post = reshape(mean(logps_post, dims=2), nx, nz)
σ_post = reshape( std(logps_post, dims=2), nx, nz)