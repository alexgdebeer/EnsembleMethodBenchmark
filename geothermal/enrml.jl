include("setup.jl")

Ne = 100
γ = 10
i_max = 10

θs, us, Ss, λs, inds = run_lm_enrml(f, g, p, L, γ, i_max, Ne)

logps_post = reduce(hcat, get_perms(p, θ) for θ ∈ eachcol(θs[:,:,end][:,inds]))

μ_post_x = reshape(mean(logps_post[1:n_blocks, :], dims=2), nx, nz)
σ_post_x = reshape( std(logps_post[1:n_blocks, :], dims=2), nx, nz)

μ_post_z = reshape(mean(logps_post[n_blocks+1:end, :], dims=2), nx, nz)
σ_post_z = reshape( std(logps_post[n_blocks+1:end, :], dims=2), nx, nz)