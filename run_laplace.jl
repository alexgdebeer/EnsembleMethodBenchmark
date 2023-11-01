using HDF5

include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

N_SAMPLES = 1000
RESULTS_FNAME = "data/laplace/laplace.h5"

η0 = vec(rand(pr, 1))
map, Γ_post, L_post = compute_laplace(grid_c, model_r, pr, d_obs, η0) # TODO: fix η -- it is returned as a sparse vector

ηs = map.η .+ L_post * rand(Normal(), pr.Nη, N_SAMPLES)
θs = hcat([transform(pr, ηi) for ηi ∈ eachcol(ηs)]...)
Fs = hcat([F(θi) for θi ∈ eachcol(θs)]...)
Gs = hcat([G(Fi) for Fi ∈ eachcol(Fs)]...)

μ_post = reshape(map.θ, grid_c.nx, grid_c.nx)
σ_post = reshape(std(θs, dims=2), grid_c.nx, grid_c.nx)

θis = θs[3200, :]
ls = [pr.l_bounds[1] + cdf(Normal(), l) * pr.Δl for l ∈ ηs[end, :]]

h5write(RESULTS_FNAME, "ηs", Matrix(ηs))
h5write(RESULTS_FNAME, "θs", θs)
h5write(RESULTS_FNAME, "Fs", model_r.B_wells * Fs)
h5write(RESULTS_FNAME, "Gs", Gs)
h5write(RESULTS_FNAME, "μ", μ_post)
h5write(RESULTS_FNAME, "σ", σ_post)
h5write(RESULTS_FNAME, "θi", θis)
h5write(RESULTS_FNAME, "l", ls)