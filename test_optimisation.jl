using LinearAlgebra
using LinearSolve
using SparseArrays

include("setup.jl")

# TODO: verify the correctness of the Jacobian (of the forward model 
# w.r.t. θ) using finite differences

function optimise(
    g::Grid,
    pr, # Prior,
    y::AbstractVector, # Data
    Q::AbstractMatrix, # Matrix of forcing terms
    θ::AbstractVector, # Initial estimate of θ
    Γ_ϵ_inv::AbstractMatrix # Covariance of errors
)

    function compute_DGθ(
        θ::AbstractVector, 
        u::AbstractVector
    )::AbstractMatrix

        u = reshape(u, g.nx^2, g.nt)

        DGθ = (g.Δt / g.μ) * vcat([
            g.∇h' * spdiagm(g.∇h * u[:, t]) * 
            spdiagm((g.A * exp.(-θ)).^-2) * g.A *
            spdiagm(exp.(-θ)) for t ∈ 1:g.nt
        ]...)

        return DGθ

    end

    function solve_adjoint(
        u::AbstractVector, 
        Bθt::AbstractMatrix
    )::AbstractVector

        p = spzeros(g.nx^2, g.nt)

        b = -g.B' * Γ_ϵ_inv * sparsevec(g.B * u - y) 
        b = reshape(b, g.nx^2, g.nt)

        prob = LinearProblem(Bθt, b[:, end])
        p[:, end] = solve(prob)

        for t ∈ (g.nt-1):-1:1
            prob = LinearProblem(Bθt, b[:, t] + g.ϕ * g.c * p[:, t+1])
            p[:, t] = solve(prob)
        end

        return Vector(vec(p))

    end
    
    function compute_∇Lθ(
        θ::AbstractVector, 
        p::AbstractVector, 
        DGθt::AbstractMatrix
    )::AbstractVector

        return pr.Γ_inv * (θ - pr.μ) + DGθt * p

    end

    function solve_forward_inc(
        Bθ::AbstractMatrix,
        b::AbstractVector
    )::AbstractVector

        b = reshape(Vector(b), g.nx^2, g.nt)
        u = spzeros(g.nx^2, g.nt)

        prob = LinearProblem(Bθ, b[:, 1])
        u[:, 1] = solve(prob)

        for t ∈ 2:g.nt 
            prob = LinearProblem(Bθ, b[:, t] + g.ϕ*g.c*u[:, t-1])
            u[:, t] = solve(prob)
        end

        return sparsevec(u)

    end

    function solve_adjoint_inc(
        Bθt::AbstractMatrix,
        b::AbstractVector 
    )::AbstractVector

        b = reshape(b, g.nx^2, g.nt)
        p = spzeros(g.nx^2, g.nt)

        prob = LinearProblem(Bθt, b[:, end])
        p[:, end] = solve(prob)

        for t ∈ (g.nt-1):-1:1
            prob = LinearProblem(Bθt, b[:, t] + g.ϕ * g.c * p[:, t+1])
            p[:, t] = solve(prob)
        end

        return Vector(vec(p))

    end

    function compute_Hd(
        d::AbstractVector, 
        Bθ::AbstractMatrix, 
        Bθt::AbstractMatrix,
        DGθ::AbstractMatrix,
        DGθt::AbstractMatrix
    )::AbstractVector

        # Solve incremental forward and adjoint problems
        u_inc = solve_forward_inc(Bθ, DGθ * d)
        p_inc = solve_adjoint_inc(Bθt, g.B' * Γ_ϵ_inv * g.B * u_inc)

        return DGθt * p_inc + pr.Γ_inv * d

    end

    # Convergence parameters for CG 
    ϵ = 0.05
    i_max = 1_000

    # TODO: convergence test
    while true
        
        # 0. Form Aθ and Bθ at the current estimate of θ
        Aθ = (1.0 / g.μ) * g.∇h' * spdiagm((g.A * exp.(-θ)) .^ -1) * g.∇h
        Bθ = g.ϕ * g.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * Aθ
        Bθt = sparse(Bθ')

        # 1. Solve forward problem for u
        u = solve(g, θ, Q)

        # 1.1. Compute gradient of ̃A(θ)u at current estimate of θ, u
        DGθ = compute_DGθ(θ, u)
        DGθt = @time sparse(DGθ')

        # 2. Solve adjoint problem for p 
        p = solve_adjoint(u, Bθt)

        # 3. Form gradient of Lagrangian w.r.t. θ
        ∇Lθ = compute_∇Lθ(θ, p, DGθt)

        # Begin inner CG loop

        # Start δθ at 0 in the absence of the better guess
        δθ = spzeros(g.nx^2)

        # Define initial search direction and residual
        d = -copy(∇Lθ)
        r = -copy(∇Lθ)

        i = 1
        while true
            
            Hd = compute_Hd(d, Bθ, Bθt, DGθ, DGθt)
            
            # Compute step length and take a step 
            α = (r' * r) / (d' * Hd)
            δθ += α * Hd 

            # Update residual vector
            r_prev = copy(r)
            r -= α * Hd

            @info "Iteration $i. ||r||^2: $(round(r'*r))"
            @info "Squared norm of residual: $(r' * r)"

            if (r' * r < ϵ^2 * ∇Lθ' * ∇Lθ) || (i > i_max)
                @info "Converged..."
                break
            end
            
            # Compute new search direction
            β = (r' * r) / (r_prev' * r_prev)
            d = r + β * d

            i += 1

        end

        # Form new estimate of θ
        θ += δθ
        display(θ)

    end

    return θ

end

optimise(grid_c, pr, y_obs, Q_c, vec(rand(pr, 1)), Γ_ϵ_inv)