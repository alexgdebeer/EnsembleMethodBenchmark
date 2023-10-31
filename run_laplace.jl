include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

N_samp = 1000
RESULTS_FNAME = "data/laplace/laplace.h5"

η0 = vec(rand(pr, 1))
map, Γ_post, L_post = compute_laplace(grid_c, model_r, pr, d_obs, η0)

ηs = map.η .+ Matrix(L_post) * rand(Normal(), pr.Nη, N_samp)
θs = hcat([transform(pr, ηi) for ηi ∈ eachcol(ηs)]...)
Fs = hcat([F(θi) for θi ∈ eachcol(θs)]...)
Gs = hcat([G(Fi) for Fi ∈ eachcol(Fs)]...)

μ_post = reshape(copy(map.θ), grid_c.nx, grid_c.nx)
σ_post = reshape(std(θs, dims=2), grid_c.nx, grid_c.nx)

h5write(RESULTS_FNAME, "ηs", ηs)
h5write(RESULTS_FNAME, "θs", θs)
h5write(RESULTS_FNAME, "Fs", model_r.B_wells * Fs)
h5write(RESULTS_FNAME, "Gs", Gs)
h5write(RESULTS_FNAME, "μ_post", μ_post)
h5write(RESULTS_FNAME, "σ_post", σ_post)

# heatmap(grid_c.xs, grid_c.xs, reshape(θ_samp[:, 1], grid_c.nx, grid_c.nx), cmap=:turbo, aspect_ratio=:equal, clims=extrema(θ_t))