using ForwardDiff
using SimIntensiveInference

include("problem_setup.jl")

L_θ = cholesky(inv(π.Σ)).U
L_ϵ = cholesky(inv(L.Σ)).U

logps_map = SimIntensiveInference.calculate_map(f, g, π, L, L_θ, L_ϵ; θ_s=vec(logps_t))
# logps_map = logps_t

J = ForwardDiff.jacobian(x->g(f(x)), vec(logps_t))

# Compute posterior covariance using Laplace approximation
Γ_post = inv(J' * inv(L.Σ) * J + inv(π.Σ))

post_stds = reshape(sqrt.(diag(Γ_post)), g_c.nx, g_c.ny)