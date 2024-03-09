include("setup.jl")

fname = "data/prior/prior.h5"
n_samples = 1000

θs = rand(pr, n_samples)
us = hcat([transform(pr, θ_i) for θ_i ∈ eachcol(θs)]...)
Fs = model_r.B_wells * hcat([F(u_i) for u_i ∈ eachcol(us)]...)
ls = [gauss_to_unif(ω_σ, σ_bounds...) for ω_σ ∈ θs[end-1, :]]

μ_θ = mean(θs, dims=2)
μ_pri = reshape(transform(pr, μ_θ), grid_c.nx, grid_c.nx)
σ_pri = reshape(std(us, dims=2), grid_c.nx, grid_c.nx)

h5write(fname, "mean", μ_pri)
h5write(fname, "stds", σ_pri)
h5write(fname, "θs", θs)
h5write(fname, "us", us)
h5write(fname, "Fs", Fs)
h5write(fname, "ls", ls)