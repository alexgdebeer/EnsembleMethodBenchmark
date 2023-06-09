using ForwardDiff
using SimIntensiveInference

include("problem_setup.jl")

L_θ = cholesky(inv(π.Σ)).U
L_ϵ = cholesky(inv(L.Σ)).U

θ_map = SimIntensiveInference.calculate_map(f, g, π, L, L_θ, L_ϵ; x0=θs_t)

J = ForwardDiff.jacobian(x -> g(f(x)), θ_map)

# Compute posterior covariance using Laplace approximation
Γ_post = inv(J' * inv(L.Σ) * J + inv(π.Σ))

post = MvNormal(θ_map, Γ_post)

plot_approx_posterior(
    eachcol(rand(post, 10_000)), 
    as, bs, post_marg_a, post_marg_b, 
    "Laplace Approximation",
    "$(plots_dir)/laplace_approx.pdf",
    θs_t=θs_t
)