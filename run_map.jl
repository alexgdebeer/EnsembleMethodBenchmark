include("setup.jl")
include("InferenceAlgorithms/map.jl")

η = vec(rand(pr, 1))
η = vec(rand(pr, 1))
η_map, u_map = compute_map(grid_c, pr, y_obs, Q_c, η, μ_u, V_r, μ_ε, Γ_e_inv)

θ_map = reshape(transform(pr, η_map), grid_c.nx, grid_c.nx)