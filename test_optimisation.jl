
# How to carry out optimisation using the adjoint method?

# Compute gradients of the forward problem with respect to u, and with 
# respect to θ

# NOTE: the prior covariance is just the identity

include("setup.jl")

# Generate a draw from the prior
θ = vec(rand(p, 1))
lnps = transform(p, θ)

us = @time solve(grid_c, lnps, Q_c)
us = reshape(us, grid_c.nx^2, grid_c.nt)

@time Gs = grid_c.Δt * vcat([
    grid_c.∇h' * spdiagm(grid_c.∇h * us[:, i]) * 
    spdiagm((grid_c.A * exp.(-(lnp_mu .+ p.A * θ))).^-2) * grid_c.A * p.A *
    spdiagm(exp.(-(lnp_mu .+ p.A * θ))) for i ∈ 1:grid_c.nt
]...)

Δθ = 0.01

for θ