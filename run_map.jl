include("setup.jl")
include("InferenceAlgorithms/map.jl")

η0 = vec(rand(pr, 1))
map, Γ_post, L_post = compute_laplace(grid_c, model_r, pr, d_obs, η0)

# L_post = cholesky(Hermitian(inv(Hermitian(H_GN) + I))).L

N_samp = 1000
η_samp = map.η .+ L_post * rand(Normal(), pr.Nη, N_samp)
θ_samp = hcat([transform(pr, η) for η ∈ eachcol(η_samp)]...)

stds = reshape(std(θ_samp, dims=2), grid_c.nx, grid_c.nx)

heatmap(grid_c.xs, grid_c.xs, reshape(θ_samp, grid_c.nx, grid_c.nx), cmap=:turbo, aspect_ratio=:equal, clims=extrema(θ_t))