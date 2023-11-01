using HDF5

include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

RESULTS_FNAME = "data/eki/eki.h5"

Ne = 1000
ηs, θs, Fs, Gs, αs = run_eki_dmc(F, G, pr, d_obs, μ_e, Γ_e, L_e, Ne)

μ_post_η = mean(ηs[end], dims=2)
μ_post = reshape(transform(pr, μ_post_η), grid_c.nx, grid_c.nx)
σ_post = reshape(std(θs[end], dims=2), grid_c.nx, grid_c.nx)

θis = θs[end][3200, :]
ls = [pr.l_bounds[1] + cdf(Normal(), l) * pr.Δl for l ∈ ηs[end][end, :]]

h5write(RESULTS_FNAME, "ηs", ηs[end])
h5write(RESULTS_FNAME, "θs", θs[end])
h5write(RESULTS_FNAME, "Fs", model_r.B_wells * Fs[end])
h5write(RESULTS_FNAME, "Gs", Gs[end])
h5write(RESULTS_FNAME, "αs", [Float64(α) for α ∈ αs])
h5write(RESULTS_FNAME, "μ", μ_post)
h5write(RESULTS_FNAME, "σ", σ_post)
h5write(RESULTS_FNAME, "θi", θis)
h5write(RESULTS_FNAME, "l", ls)