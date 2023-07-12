include("setup.jl")

Ni = 8
Ne = 100
θs, fs, us, αs, inds = run_es_mda(f, g, p, L, n_blocks, Ni, Ne)#, α_method=:constant)

logps_post = reduce(hcat, get_perms(p, θ) for θ ∈ eachcol(θs[:,inds,end]))

μ_post = reshape(mean(logps_post, dims=2), nx, nz)
σ_post = reshape( std(logps_post, dims=2), nx, nz)