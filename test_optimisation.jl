using LinearAlgebra
using SparseArrays

include("setup.jl")

# How to carry out optimisation using the adjoint method?

# Compute gradients of the forward problem with respect to u, and with 
# respect to θ

# NOTE: the prior covariance is just the identity

# Generate a starting point from the prior
lnps = vec(rand(p, 1))

us = @time solve(grid_c, lnps, Q_c)
us = reshape(us, grid_c.nx^2, grid_c.nt)

# Form Jacobian of forward model with respect to k
DGk = (grid_c.Δt / grid_c.μ) * vcat([
    grid_c.∇h' * spdiagm(grid_c.∇h * us[:, i]) * 
    spdiagm((grid_c.A * exp.(-lnps)).^-2) * grid_c.A *
    spdiagm(exp.(-lnps)) for i ∈ 1:grid_c.nt
]...)

# Form Jacobian of forward model with respect to u
B = (grid_c.ϕ * grid_c.c) * sparse(I, grid_c.nx^2, grid_c.nx^2) + (grid_c.Δt / grid_c.μ) * grid_c.∇h' * spdiagm((grid_c.A * exp.(-lnps)).^-1) * grid_c.∇h

DGu = blockdiag([B for _ ∈ 1:grid_c.nt]...)
DGu[(grid_c.nx^2+1):end, 1:(end-grid_c.nx^2)] -= (grid_c.ϕ * grid_c.c) * sparse(I, (grid_c.nx)^2*(grid_c.nt-1), (grid_c.nx)^2*(grid_c.nt-1))

# Δθ = 0.01

# TODO: figure out how to do this with a reduced-order model 
