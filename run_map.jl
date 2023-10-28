include("setup.jl")
include("InferenceAlgorithms/map.jl")

η0 = vec(rand(pr, 1))
# η_map, u_map = compute_map(grid_c, model_r, pr, d_obs, η0)

# θ_map = reshape(transform(pr, η_map), grid_c.nx, grid_c.nx)

compute_laplace(grid_c, model_r, pr, d_obs, η0)