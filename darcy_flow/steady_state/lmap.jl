using SimIntensiveInference

include("setup.jl")

L_ϵ = inv_cholesky(L.Σ)

sol = calculate_map(f, g, p, L, L_ϵ, p.L, x0=vec(θs_t))

logps_map = reshape(get_perms(p, sol.θ_min), grid.nx, grid.ny)
J = inv(L_ϵ) * sol.J_min[1:length(L.μ), :]

# Compute posterior covariance using Laplace approximation
Γ_post = inv(J' * inv(L.Σ) * J + inv(p.Γ))

using ForwardDiff, FiniteDiff

residuals(θ) = [
    L_ϵ * (g(f(θ)) - L.μ); 
    p.L * (θ - p.μ)
]

J_forward = @time ForwardDiff.jacobian(residuals, sol.θ_min)
J_finite = @time FiniteDiff.finite_difference_jacobian(residuals, sol.θ_min, absstep=0.1)

# σ_post = reshape(sqrt.(diag(Γ_post)), grid.nx, grid.ny)

# fname = "plots/darcy_flow/laplace/laplace_mean.pdf"
# plot_μ_post(logps_map, logps_t, g_c, g_f, fname, title="MAP estimate")

# fname = "plots/darcy_flow/laplace/laplace_stds.pdf"
# plot_σ_post(σ_post, g_c, fname)