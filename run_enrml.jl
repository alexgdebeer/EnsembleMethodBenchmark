using HDF5

include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

RESULTS_FNAME = "data/enrml/enrml.h5"

Ne = 100
NF = grid_c.nx^2 * grid_c.nt
i_max = 20
ηs, θs, Fs, Gs, λs = run_enrml(F, G, pr, d_obs, μ_e, Γ_e, Ne, NF, i_max)

μ_post_η = mean(ηs[:, :, end], dims=2)
μ_post = reshape(transform(pr, μ_post_η), grid_c.nx, grid_c.nx)
σ_post = reshape(std(θs[:, :, end], dims=2), grid_c.nx, grid_c.nx)