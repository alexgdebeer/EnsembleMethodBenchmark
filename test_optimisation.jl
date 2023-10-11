using LinearAlgebra
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

        DGθ = g.Δt * vcat([
            (1.0 / g.μ) * g.∇h' * spdiagm(g.∇h * u[:, t]) * 
            spdiagm((g.A * exp.(-θ)).^-2) * g.A *
            spdiagm(exp.(-θ)) for t ∈ 1:g.nt
        ]...)

        return DGθ

    end

    function solve_adjoint(
        u::AbstractVector, 
        Bθ::AbstractMatrix
    )::AbstractVector

        p = spzeros(g.nx^2, g.nt)

        b = g.B' * Γ_ϵ_inv * sparsevec(g.B * u - y) 
        b = reshape(b, g.nx^2, g.nt)

        # TODO: figure out why \ works but forming a LinearProblem doesn't
        p[:, end] = Bθ' \ Vector(b[:, end])

        for t ∈ (g.nt-1):-1:1

            bt = b[:, t] + g.ϕ * g.c * p[:, t+1]
            p[:, t] = Bθ' \ Vector(bt)

        end

        return vec(p)

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
    )::AbstractVector

        b = reshape(Vector(b), g.nx^2, g.nt)
        u = spzeros(g.nx^2, g.nt)

        u[:, 1] = Bθ \ b[:, 1]

        for t ∈ 2:g.nt 

            bt = b[:, t] + g.ϕ * g.c * u[:, t-1]
            u[:, t] = Bθ \ Vector(bt)

        end

        return vec(u)

    end

    function solve_adjoint_inc(
        Bθ::AbstractMatrix,
        b::AbstractVector 
    )::AbstractVector

        b = reshape(Vector(b), g.nx^2, g.nt)
        p = spzeros(g.nx^2, g.nt)

        p[:, end] = Bθ' \ b[:, end]

        # Backward solve this time because the A matrix is transposed
        for t ∈ (g.nt-1):-1:1

            bt = b[:, t] + g.ϕ * g.c * p[:, t+1]
            p[:, t] = Bθ' \ Vector(bt)

        end

        return sparsevec(p)

    end

    function compute_Hd(
        d::AbstractVector, 
        Bθ::AbstractMatrix, 
        DGθ::AbstractMatrix 
    )::AbstractVector

        # Solve incremental forward and adjoint problems
        u_inc = solve_forward_inc(Bθ, DGθ * d)
        p_inc = solve_adjoint_inc(Bθ, g.B' * Γ_ϵ_inv * g.B * u_inc)

        return DGθ' * p_inc + pr.Γ_inv * d

    end

    # Convergence parameters for CG 
    ϵ = 0.05
    i_max = 1_000

    # TODO: convergence test
    while true
        
        # 0. Form Aθ and Bθ at the current estimate of θ
        Aθ = (1.0 / g.μ) * g.∇h' * spdiagm((g.A * exp.(θ)) .^ -1) * g.∇h
        Bθ = g.ϕ * g.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * Aθ

        # 1. Solve forward problem for u
        u = @time solve(g, θ, Q)

        # 1.1. Compute gradient of ̃A(θ)u at current estimate of θ, u
        DGθ = compute_DGθ(θ, u)

        # 2. Solve adjoint problem for p 
        p = solve_adjoint(u, Bθ)

        # 3. Form gradient of Lagrangian w.r.t. θ
        ∇Lθ = @time compute_∇Lθ(θ, p, DGθ)

        # Begin inner CG loop

        # Start δθ at 0 in the absence of the better guess
        δθ = spzeros(g.nx^2)

        # Define initial search direction and residual
        d = -copy(∇Lθ)
        r = -copy(∇Lθ)

        display(u)
        display(DGθ)
        display(p)
        display(∇Lθ)

        i = 1
        while true
            
            Hd = @time compute_Hd(d, Bθ, DGθ)
            
            # Compute step length and take a step 
            α = (r' * r) / (d' * Hd)
            δθ += α * Hd 

            # Update residual vector
            r_prev = copy(r)
            r -= α * Hd

            @info "Iteration $i."
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

    end

    return θ

end

optimise(grid_c, pr, y_obs, Q_c, vec(rand(pr, 1)), Γ_ϵ_inv)