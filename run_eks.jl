using HDF5

include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

RESULTS_FNAME = "data/eks/eks.h5"

Ne = 10_000
ηs, θs, Fs, Gs = run_eks(F, G, pr, d_obs, μ_e, Γ_e, Ne)

μ_post_η = mean(ηs[end], dims=2)
μ_post = reshape(transform(pr, μ_post_η), grid_c.nx, grid_c.nx)
σ_post = reshape(std(θs[end], dims=2), grid_c.nx, grid_c.nx)