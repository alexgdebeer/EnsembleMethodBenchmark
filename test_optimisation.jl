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

# # Form Jacobian of forward model with respect to θ
# DGθ = (grid_c.Δt / grid_c.μ) * vcat([
#     grid_c.∇h' * spdiagm(grid_c.∇h * us[:, i]) * 
#     spdiagm((grid_c.A * exp.(-lnps)).^-2) * grid_c.A *
#     spdiagm(exp.(-lnps)) for i ∈ 1:grid_c.nt
# ]...)

# Form Jacobian of forward model with respect to u
B = (grid_c.ϕ * grid_c.c) * sparse(I, grid_c.nx^2, grid_c.nx^2) + (grid_c.Δt / grid_c.μ) * grid_c.∇h' * spdiagm((grid_c.A * exp.(-lnps)).^-1) * grid_c.∇h

DGu = blockdiag([B for _ ∈ 1:grid_c.nt]...)
DGu[(grid_c.nx^2+1):end, 1:(end-grid_c.nx^2)] -= (grid_c.ϕ * grid_c.c) * sparse(I, (grid_c.nx)^2*(grid_c.nt-1), (grid_c.nx)^2*(grid_c.nt-1))

# TODO: verify the Jacobian using finite differences


function optimise(
    g::Grid,
    pr, # Prior,
    y::AbstractVector, # Data
    Q::AbstractMatrix, # Matrix of forcing terms
    B::AbstractMatrix, # Observation operator
    θ::AbstractVector, # Initial estimate of θ
    Γϵ_inv::AbstractMatrix # Covariance of errors
)

    function compute_DGθ(
        θ::AbstractVector, 
        u::AbstractMatrix
    )::AbstractMatrix

        DGθ = g.Δt * vcat([
            (1.0 / g.μ) * g.∇h' * spdiagm(g.∇h * u[:, i]) * 
            spdiagm((g.A * exp.(-θ)).^-2) * g.A *
            spdiagm(exp.(-θ)) for i ∈ 1:g.nt
        ]...)

        return DGθ

    end

    function compute_p(
        u::AbstractVector, 
        Bθ::AbstractMatrix
    )::AbstractVector

        p = spzeros(g.nx^2, g.nt)

        # TODO: currently assuming no observations at the first time -- should I?

        for i ∈ 2:g.nt
            
            # TODO: define t_inds somewhere
            if i ∈ t_inds 
                
                # TODO: figure out how y should be indexed
                # TODO: Maybe just work it all out in one go with a big B matrix
                b = B' * Γϵ_inv * (B * u[:, i] - y[:, -1]) 
                p[:, i] = solve(Bθ, b)

            end

        end

        return p

    end
    
    function compute_∇Lθ(
        θ::AbstractVector, 
        p::AbstractVector, 
        DGθ::AbstractMatrix
    )::AbstractVector

        return pr.Γ_inv * (θ - pr.μ) + DGθ' * p

    end

    function solve_forward_inc(
        Bθ::AbstractMatrix,
        b::AbstractVector
    )

        b = reshape(b, g.nx^2, g.nt)
        u = spzeros(g.nx^2, g.nt)

        u[:, 1] = solve(LinearProblem(Bθ, b[:, 1]))

        for t ∈ 2:g.nt 

            bt = b[:, t] + g.ϕ * g.c * u[:, t-1]
            u[:, t] = solve(LinearProblem(A, bt))

        end

        return u

    end

    function solve_adjoint_inc(
        Bθ::AbstractMatrix,
        b::AbstractVector 
    )

        b = reshape(b, g.nx^2, g.nt)
        p = spzeros(g.nx^2, g.nt)

        p[:, end] = solve(LinearProblem(Bθ', b[:, end]))

        # Backward solve this time because the A matrix is transposed
        for t ∈ (g.nt-1):-1:1

            bt = b[:, t] + g.ϕ * g.c * p[:, t+1]
            p[:, t] = solve(LinearProblem(Bθ', bt))

        end

        return p

    end

    function compute_Hd(
        d::AbstractVector, 
        Bθ::AbstractMatrix, 
        DGθ::AbstractMatrix 
    )::AbstractVector

        # Solve incremental forward and adjoint problems
        # TODO: fix the B matrix (it is for one timestep only, but 
        # should be for all of them)
        u_inc = solve_forward_inc(Bθ, DGθ * d)
        p_inc = solve_adjoint_inc(Bθ, B' * pr.Γ_inv * B * u_inc)

        return G * p_inc + pr.Γ_inv

    end

    # TODO: should I include the initial condition in the u vector 
    # that is getting returned?

    # TODO: build the complete observation operator

    # TODO: convergence test
    while true

        # 0. Form Aθ and Bθ at the current estimate of θ
        Aθ = (1.0 / g.μ) * g.∇h' * spdiagm((g.A * exp.(θ)) .^ -1) * g.∇h
        Bθ = g.ϕ * g.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * Aθ

        # 1. Solve forward problem for u (TODO: put into here...?)
        u = solve(g, θ, Q)

        # 1.1. Compute gradient of ̃A(θ)u at current estimate of θ, u
        DGθ = compute_DGθ(θ, u)

        # 2. Solve adjoint problem for p 
        p = compute_p(θ, Bθ)

        # 3. Form gradient of Lagrangian w.r.t. θ
        ∇Lθ = compute_∇Lθ(θ, p, DGθ)

        # Begin inner CG loop 

        # Start δθ at 0 in the absence of the better guess
        δθ = spzeros(g.nx^2)

        # Define initial search direction and residual
        d = -∇Lθ
        r = -∇Lθ

        # TODO: convergence test
        # TODO: form big A matrix
        while true
            
            Hd = compute_Hd(d, Bθ, DGθ)
            
            # Compute step length and take a step 
            α = (r' * r) / (d' * Hδθ * d)
            δθ += α * Hd 

            # Update residual vector
            r_prev = copy(r)
            r -= α*Ad
            
            # Compute new search direction
            β = (r' * r) / (r_prev' * r_prev)
            d = r + β * d

        end

    end

end