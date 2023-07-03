using SimIntensiveInference

include("setup.jl")

L_ϵ = inv_cholesky(L.Σ)

sol = calculate_map(f, g, p, L, L_ϵ, p.L)#, x0=vec(logps_t))

logps_map = sol.θ_min
J = inv(L_ϵ) * sol.J_min[1:length(L.μ), :]

# Compute posterior covariance using Laplace approximation
Γ_post = inv(J' * inv(L.Σ) * J + inv(p.Γ))

σ_post = reshape(sqrt.(diag(Γ_post)), grid_c.nx, grid_c.ny)

logps_map = reshape(logps_map, grid_c.nx, grid_c.ny)