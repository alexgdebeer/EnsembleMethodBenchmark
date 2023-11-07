using HDF5

include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

RESULTS_FNAME = "data/enrml/enrml.h5"

Ne = 1000
NF = grid_c.nx^2 * grid_c.nt
i_max = 20
ηs, θs, Fs, Gs, λs = run_enrml(F, G, pr, d_obs, μ_e, Γ_e, Ne, NF, i_max)

μ_post_η = mean(ηs[:, :, end], dims=2)
μ_post = reshape(transform(pr, μ_post_η), grid_c.nx, grid_c.nx)
σ_post = reshape(std(θs[:, :, end], dims=2), grid_c.nx, grid_c.nx)

h5write(RESULTS_FNAME, "ηs", ηs[:, :, end])
h5write(RESULTS_FNAME, "θs", θs[:, :, end])
h5write(RESULTS_FNAME, "Fs", model_r.B_wells * Fs[:, :, end])
h5write(RESULTS_FNAME, "Gs", Gs[:, :, end])
h5write(RESULTS_FNAME, "μ", μ_post)
h5write(RESULTS_FNAME, "σ", σ_post)