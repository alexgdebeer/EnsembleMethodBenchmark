using SimIntensiveInference

include("problem_setup.jl")

L_ϵ = inv_cholesky(L.Σ)
L_θ = inv_cholesky(π.Σ)

sol = calculate_map(f, g, π, L, L_ϵ, L_θ; x0=θs_t)

J = inv(L_ϵ) * sol.J_min[1:length(L.μ), :]

# Compute posterior covariance using Laplace approximation
Γ_post = inv(J' * inv(L.Σ) * J + inv(π.Σ))

post = MvNormal(sol.θ_min, Γ_post)

plot_approx_posterior(
    eachcol(rand(post, 10_000)), 
    as, bs, post_marg_a, post_marg_b, 
    "Laplace Approximation",
    "$(plots_dir)/laplace_approx.pdf",
    θs_t=θs_t
)