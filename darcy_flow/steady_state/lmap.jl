using SimIntensiveInference

include("setup.jl")

L_ϵ = inv_cholesky(L.Σ)

sol = calculate_map(f, g, p, L, L_ϵ, p.L)

logps_map = sol.θ_min
J = inv(L_ϵ) * sol.J_min[1:length(L.μ), :]

# Compute posterior covariance using Laplace approximation
Γ_post = inv(J' * inv(L.Σ) * J + inv(p.Γ))

σ_post = reshape(sqrt.(diag(Γ_post)), grid.nx, grid.ny)

logps_map = reshape(logps_map, grid.nx, grid.ny)

# fname = "plots/darcy_flow/laplace/laplace_mean.pdf"
# plot_μ_post(logps_map, logps_t, g_c, g_f, fname, title="MAP estimate")

# fname = "plots/darcy_flow/laplace/laplace_stds.pdf"
# plot_σ_post(σ_post, g_c, fname)