using SimIntensiveInference

include("problem_setup.jl")
include("plotting.jl")

L_ϵ = inv_cholesky(L.Σ)
L_θ = inv_cholesky(π.Σ)

sol = calculate_map(f, g, π, L, L_ϵ, L_θ)

logps_map = sol.θ_min
J = inv(L_ϵ) * sol.J_min[1:length(L.μ), :]

# Compute posterior covariance using Laplace approximation
Γ_post = inv(J' * inv(L.Σ) * J + inv(π.Σ))

σ_post = reshape(sqrt.(diag(Γ_post)), g_c.nx, g_c.ny)

logps_map = reshape(logps_map, g_c.nx, g_c.ny)

fname = "plots/darcy_flow/laplace/laplace_mean.pdf"
plot_μ_post(logps_map, logps_t, g_c, g_f, fname, title="MAP estimate")

fname = "plots/darcy_flow/laplace/laplace_stds.pdf"
plot_σ_post(σ_post, g_c, fname)