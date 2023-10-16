using LinearAlgebra
using LinearSolve
using SparseArrays

include("setup.jl")

using FiniteDiff # TEMP

function optimise(
    g::Grid,
    pr::MaternField,
    y::AbstractVector,
    Q::AbstractMatrix,
    η::AbstractVector, # Initial estimate of η
    Γ_ϵ_inv::AbstractMatrix
)

    # TODO: move to MaternField struct?
    Δσ = pr.σ_bounds[2] - pr.σ_bounds[1]
    Δl = pr.l_bounds[2] - pr.l_bounds[1]

    # Convergence parameters for CG 
    ϵ = 1e-8
    j_max = 20

    function compute_∂Au∂θ(
        θ::AbstractVector, 
        u::AbstractVector
    )::AbstractMatrix

        u = reshape(u, g.nx^2, g.nt)

        ∂Au∂θ = (g.Δt / g.μ) * vcat([
            g.∇h' * spdiagm(g.∇h * u[:, t]) * 
            g.A * spdiagm(exp.(θ)) for t ∈ 1:g.nt
        ]...)

        return ∂Au∂θ

    end

    function solve_adjoint(
        u::AbstractVector, 
        Bθt::AbstractMatrix
    )::AbstractVector

        p = spzeros(g.nx^2, g.nt)

        b = -g.B' * Γ_ϵ_inv * (g.B * u - y) 
        b = reshape(b, g.nx^2, g.nt)

        prob = LinearProblem(Bθt, b[:, end])
        p[:, end] = solve(prob)

        for t ∈ (g.nt-1):-1:1
            prob = LinearProblem(Bθt, b[:, t] + g.ϕ * g.c * p[:, t+1])
            p[:, t] = solve(prob)
        end

        return Vector(vec(p))

    end
    
    function compute_∂Au∂ηtx(∂Au∂θt, η, θ, x)

        ξ_σ, ξ_l = η[end-1:end]

        σ = gauss_to_unif(ξ_σ, pr.σ_bounds...)
        l = gauss_to_unif(ξ_l, pr.l_bounds...)

        α = σ^2 * (4π * gamma(2)) / gamma(1)

        H = pr.M + l^2 * pr.K + l / 1.42 * pr.N

        ∂Au∂θtx = ∂Au∂θt * x

        # White noise component 
        ∂Au∂ξtx = √(α) * l * pr.L' * solve(LinearProblem(H, ∂Au∂θtx))

        # Standard deviation component
        ∂Au∂σtx = (θ / σ)' * ∂Au∂θtx
        ∂Au∂ξσtx = Δσ * pdf(Normal(), ξ_σ) * ∂Au∂σtx
        
        # Lengthscale component
        ∂Au∂ltx = θ' * (-l^-1.0 * pr.M + l * pr.K)' * solve(LinearProblem(H, ∂Au∂θtx))
        ∂Au∂ξltx = Δl * pdf(Normal(), ξ_l) * ∂Au∂ltx

        return vcat(∂Au∂ξtx, ∂Au∂ξσtx, ∂Au∂ξltx)

    end

    function compute_∂Au∂ηx(∂Au∂θ, η, θ, x)

        ξ_σ, ξ_l = η[end-1:end]

        σ = gauss_to_unif(ξ_σ, pr.σ_bounds...)
        l = gauss_to_unif(ξ_l, pr.l_bounds...)

        α = σ^2 * (4π * gamma(2)) / gamma(1)

        H = pr.M + l^2 * pr.K + l / 1.42 * pr.N 

        # White noise component
        ∂θ∂ξx = solve(LinearProblem(H, √(α) * l * pr.L * x[1:end-2]))
        ∂Au∂ξx = ∂Au∂θ * sparsevec(∂θ∂ξx)

        # Standard deviation component
        ∂σ∂ξσx = Δσ * pdf(Normal(), ξ_σ) * x[end-1]
        ∂Au∂ξσx = ∂Au∂θ * (sparsevec(θ) / σ) * ∂σ∂ξσx

        # Lengthscale component
        ∂l∂ξlx = Δl * pdf(Normal(), ξ_l) * x[end]
        ∂θ∂ξlx = solve(LinearProblem(H, (-l^-1.0 * pr.M + l * pr.K) * θ * ∂l∂ξlx))
        ∂Au∂ξlx = ∂Au∂θ * ∂θ∂ξlx

        return ∂Au∂ξx + ∂Au∂ξσx + ∂Au∂ξlx

    end

    function compute_∇Lη(
        ∂Au∂θt::AbstractMatrix,
        η::AbstractVector,
        θ::AbstractVector,
        p::AbstractVector
    )::AbstractVector

        # Prior is whitened
        return η + compute_∂Au∂ηtx(∂Au∂θt, η, θ, p)

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
        η::AbstractVector,
        θ::AbstractVector,
        Bθ::AbstractMatrix, 
        Bθt::AbstractMatrix,
        ∂Au∂θ::AbstractMatrix,
        ∂Au∂θt::AbstractMatrix
    )::AbstractVector

        # Solve incremental forward and adjoint problems
        u_inc = solve_forward_inc(Bθ, compute_∂Au∂ηx(∂Au∂θ, η, θ, d))
        p_inc = solve_adjoint_inc(Bθt, g.B' * Γ_ϵ_inv * g.B * u_inc)

        return compute_∂Au∂ηtx(∂Au∂θt, η, θ, p_inc) + d

    end

    i = 1
    while true

        @info "Beginning outer loop: iteration $i"
        
        θ = transform(pr, η)

        # Form Aθ and Bθ at the current estimate of θ
        Aθ = (1.0 / g.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h
        Bθ = (g.ϕ * g.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * Aθ)
        Bθt = sparse(Bθ')

        # Solve forward and adoint problems
        u = solve(g, θ, Q)
        p = solve_adjoint(u, Bθt)

        # Compute Jacobian of forward problem and gradient of Lagrangian 
        # w.r.t. θ
        ∂Au∂θ = compute_∂Au∂θ(θ, u)
        ∂Au∂θt = sparse(∂Au∂θ')

        # TEMP: finite difference Jacobian 
        function A_full_u(η)

            θ = transform(pr, η)

            Aθ = (1.0 / g.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h
            Bθ = g.ϕ * g.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * Aθ

            A_full = blockdiag([Bθ for _ ∈ 1:g.nt]...)
            iix = (g.nx^2+1):g.nx^2*g.nt 
            iiy = 1:g.nx^2*(g.nt-1)
            A_full[iix, iiy] += -g.ϕ * g.c * sparse(I, g.nx^2*(g.nt-1), g.nx^2*(g.nt-1))

            return A_full * u

        end

        # J = FiniteDiff.finite_difference_jacobian(A_full_u, η)

        J = zeros(length(u), length(η))
        Δη = 0.01

        for i ∈ 1:length(η)

            η_p = copy(η)

            η_p[i] += Δη
            Au_1 = A_full_u(η_p)

            η_p[i] -= 2Δη
            Au_0 = A_full_u(η_p)

            J[:, i] = (Au_1 - Au_0) / 2Δη

        end


        ξ_σ, ξ_l = η[end-1:end]
        σ = gauss_to_unif(ξ_σ, pr.σ_bounds...)
        l = gauss_to_unif(ξ_l, pr.l_bounds...)
        α = σ^2 * (4π * gamma(2)) / gamma(1)
        H = pr.M + l^2 * pr.K + l / 1.42 * pr.N 

        invH = inv(Matrix(H))

        ∂Au∂ξ = ∂Au∂θ * invH * √(α) * l * pr.L
        ∂Au∂ξσ = ∂Au∂θ * ((θ .- pr.μ) ./ σ) * Δσ * pdf(Normal(), ξ_σ)
        ∂Au∂ξl = ∂Au∂θ * -invH * (-l^-1.0 * pr.M + l * pr.K) * (θ .- pr.μ) * Δl * pdf(Normal(), ξ_l)

        deriv = Matrix(hcat(∂Au∂ξ, ∂Au∂ξσ, ∂Au∂ξl))

        display(J[2011:2021, :])
        display(deriv[2011:2021, :])
        display(J[2011:2021, end-1] ./ deriv[2011:2021, end-1])

        error("Stop")

        ∇Lη = compute_∇Lη(∂Au∂θt, η, θ, p)

        @info "Norm of gradient: $(norm(∇Lη))"
        if norm(∇Lη) < 2 || i > 30
            return η, u
        end

        # Start δθ at 0 in the absence of the better guess
        δη = spzeros(pr.Nθ)

        # Define initial search direction and residual
        d = -copy(∇Lη)
        r = -copy(∇Lη)

        j = 1
        while true
            
            Hd = compute_Hd(d, η, θ, Bθ, Bθt, ∂Au∂θ, ∂Au∂θt)
            
            # Compute step length and take a step 
            α = (r' * r) / (d' * Hd)
            δη += α * d

            # Update residual vector
            r_prev = copy(r)
            r = r_prev - α * Hd

            if j % 1 == 0
                @info "Iteration $j. ||r||^2: $(r' * r)"
            end

            if (r' * r < ϵ^2 * ∇Lη' * ∇Lη) || (j > j_max)
                @info "Converged..."
                break
            end
            
            # Compute new search direction
            β = (r' * r) / (r_prev' * r_prev)
            d = r + β * d

            j += 1

        end

        # Form new estimate of η
        η += δη
        i += 1

    end

end

η = vec(rand(pr, 1)) # TODO: add POD basis
η_map, u_map = optimise(grid_c, pr, y_obs, Q_c, η, Γ_ϵ_inv)