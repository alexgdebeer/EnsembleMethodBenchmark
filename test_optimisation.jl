using LinearAlgebra
using LinearSolve
using Printf
using SparseArrays

include("setup.jl")

const GN_MIN_NORM = 0.1
const GN_MAX_ITS = 30

const CG_MAX_ITS = 30

# Define linesearch parameters
const LINE_C = 1e-4
const LINE_MAX_IT = 10

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

    # Define CG convergence parameters (TODO: figure out how they did this in Petra and Staedler)
    ϵ = 1e-4

    function J(η, u)
        resid = g.B * u - y
        return 0.5 * resid' * Γ_ϵ_inv * resid + 0.5 * sum(η.^2)
    end

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
    
    function compute_∂Au∂ηtx(
        ∂Au∂θt::AbstractMatrix, 
        η::AbstractVector, 
        θ::AbstractVector, 
        x::AbstractVector
    )::AbstractVector

        ξ_σ, ξ_l = η[end-1:end]

        σ = gauss_to_unif(ξ_σ, pr.σ_bounds...)
        l = gauss_to_unif(ξ_l, pr.l_bounds...)

        α = σ^2 * (4π * gamma(2)) / gamma(1)

        H = pr.M + l^2 * pr.K + l / 1.42 * pr.N

        ∂Au∂θtx = ∂Au∂θt * x

        # White noise component 
        ∂Au∂ξtx = √(α) * l * pr.L' * solve(LinearProblem(H, ∂Au∂θtx))

        # Standard deviation component
        ∂Au∂σtx = ((θ .- pr.μ) / σ)' * ∂Au∂θtx
        ∂Au∂ξσtx = Δσ * pdf(Normal(), ξ_σ) * ∂Au∂σtx
        
        # Lengthscale component
        ∂Au∂ltx = -(θ .- pr.μ)' * (-l^-1.0 * pr.M + l * pr.K)' * solve(LinearProblem(H, ∂Au∂θtx))
        ∂Au∂ξltx = Δl * pdf(Normal(), ξ_l) * ∂Au∂ltx

        return vcat(∂Au∂ξtx, ∂Au∂ξσtx, ∂Au∂ξltx)

    end

    function compute_∂Au∂ηx(
        ∂Au∂θ::AbstractMatrix, 
        η::AbstractVector, 
        θ::AbstractVector, 
        x::AbstractVector
    )::AbstractVector

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
        ∂Au∂ξσx = ∂Au∂θ * (sparsevec(θ .- pr.μ) / σ) * ∂σ∂ξσx

        # Lengthscale component
        ∂l∂ξlx = Δl * pdf(Normal(), ξ_l) * x[end]
        ∂θ∂ξlx = -solve(LinearProblem(H, (-l^-1.0 * pr.M + l * pr.K) * (θ .- pr.μ) * ∂l∂ξlx))
        ∂Au∂ξlx = ∂Au∂θ * ∂θ∂ξlx

        return ∂Au∂ξx + ∂Au∂ξσx + ∂Au∂ξlx

    end

    function compute_∇Lη(
        ∂Au∂θt::AbstractMatrix,
        η::AbstractVector,
        θ::AbstractVector,
        p::AbstractVector
    )::AbstractVector

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

    function linesearch(
        η_c::AbstractVector, 
        u_c::AbstractVector, 
        ∇Lη_c::AbstractVector, 
        δη::AbstractVector
    )::Real

        println("LS It. | J(η, u)")
        J_c = J(η_c, u_c)
        α_k = 1.0

        n_ls = 1
        while n_ls < LINE_MAX_IT

            η_k = η_c + α_k * δη
            θ_k = transform(pr, η_k)
            u_k = solve(g, θ_k, Q)
            J_k = J(η_k, u_k)

            @printf "%6i | %.3e\n" n_ls J_k 

            if (J_k ≤ J_c + LINE_C * α_k * ∇Lη_c' * δη)
                println("Linesearch converged after $n_ls iterations.")
                return α_k
            end

            α_k *= 0.5
            n_ls += 1

        end

        @warn "Linesearch failed to converge within $LINE_MAX_IT iterations."
        return α_k

    end

    n_gn = 1
    while true

        @info "Beginning GN It. $n_gn"
        
        θ = transform(pr, η)

        Aθ = (1.0 / g.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h
        Bθ = sparse(g.ϕ * g.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * Aθ)
        Bθt = sparse(Bθ')

        u = solve(g, θ, Q)
        p = solve_adjoint(u, Bθt)

        ∂Au∂θ = compute_∂Au∂θ(θ, u)
        ∂Au∂θt = sparse(∂Au∂θ')

        ∇Lη = compute_∇Lη(∂Au∂θt, η, θ, p)
        @printf "norm(∇Lη): %.3e\n" norm(∇Lη)

        if norm(∇Lη) < GN_MIN_NORM
            return η, u
        elseif n_gn > GN_MAX_ITS
            @warn "Gauss-Newton failed to converge within $GN_MAX_ITS iterations."
            return η, u
        end

        println("CG It. | norm(r)")
        δη = spzeros(pr.Nθ)
        d = -copy(∇Lη)
        r = -copy(∇Lη)

        n_cg = 1
        while true
            
            Hd = compute_Hd(d, η, θ, Bθ, Bθt, ∂Au∂θ, ∂Au∂θt)
            
            α = (r' * r) / (d' * Hd)
            δη += α * d

            r_prev = copy(r)
            r = r_prev - α * Hd

            @printf "%6i | %.3e\n" n_cg norm(r)

            if (norm(r) < ϵ^2 * norm(∇Lη))
                println("CG converged after $n_cg iterations.")
                break
            elseif n_cg > CG_MAX_ITS
                @warn "CG failed to converge within $CG_MAX_ITS iterations."
            end
            
            β = (r' * r) / (r_prev' * r_prev)
            d = r + β * d

            n_cg += 1

        end

        α = linesearch(η, u, ∇Lη, δη)
    
        η += α * δη
        i += 1

    end

end


η = vec(rand(pr, 1)) # TODO: add POD basis
η_map, u_map = optimise(grid_c, pr, y_obs, Q_c, η, Γ_ϵ_inv)