using LinearAlgebra
using LinearSolve
using Printf
using SparseArrays

const GN_MIN_NORM = 1e-2
const GN_MAX_ITS = 30

const CG_MAX_ITS = 30

const LS_C = 1e-4
const LS_MAX_ITS = 20

function compute_map(
    g::Grid,
    pr::MaternField,
    y::AbstractVector,
    Q::AbstractMatrix,
    η::AbstractVector,          # Initial estimate of η
    μ_uk::AbstractVector,        # Mean of u, estimated using samples
    V_rk::AbstractMatrix,        # Reduced basis for u
    μ_ε::AbstractVector,        # Mean of model errors 
    Γ_e_inv::AbstractMatrix     # Inverse of combined measurement and model error covariance
)

    # Get the size of the reduced state vector
    nu_r = size(V_rk, 2)

    # TODO: tidy this up (and the above line of code)
    V_r = sparse(kron(sparse(I, g.nt, g.nt), V_rk))
    BV_r = g.B * V_r
    μ_u = repeat(μ_uk, g.nt)

    function J(
        η::AbstractVector, 
        u::AbstractVector
    )::Real
        res = g.B * (V_r * u + μ_u) + μ_ε - y
        return 0.5 * res' * Γ_e_inv * res + 0.5 * sum(η.^2)
    end

    function compute_∂Au∂θ(
        θ::AbstractVector, 
        u::AbstractVector
    )::AbstractMatrix

        u = reshape(u, nu_r, g.nt)

        ∂Au∂θ = (g.Δt / g.μ) * vcat([
            V_rk' * g.∇h' * spdiagm(g.∇h * V_rk * u[:, t]) * 
            g.A * spdiagm(exp.(θ)) for t ∈ 1:g.nt
        ]...)

        return ∂Au∂θ

    end

    function compute_∂Aμ∂θ(
        θ::AbstractVector
    )::AbstractMatrix 

        ∂Aμ∂θ = (g.Δt / g.μ) * vcat([
            V_rk' * g.∇h' * spdiagm(g.∇h * μ_uk) * 
            g.A * spdiagm(exp.(θ)) for t ∈ 1:g.nt
        ]...)

        return ∂Aμ∂θ

    end

    function solve_forward(
        Aθ::AbstractMatrix,
        Aθ_r::AbstractMatrix
    )::AbstractVector

        u = zeros(nu_r, g.nt)

        b = V_rk' * (g.Δt * Q[:, 1] .+ (g.c * g.ϕ * u0) .- Aθ * μ_uk)
        u[:, 1] = solve(LinearProblem(Aθ_r, b))

        for t ∈ 2:g.nt 
            b = V_rk' * (g.Δt * Q[:, t] + g.ϕ * g.c * (V_rk * u[:, t-1] + μ_uk) - Aθ * μ_uk)
            u[:, t] = solve(LinearProblem(Aθ_r, b))
        end

        return vec(u)

    end

    function solve_adjoint(
        u::AbstractVector, 
        Aθ_r::AbstractMatrix
    )::AbstractVector
        
        p = zeros(nu_r, g.nt) 

        b = -BV_r' * Γ_e_inv * (BV_r * u + g.B * μ_u + μ_ε - y)
        b = reshape(b, nu_r, g.nt)

        prob = LinearProblem(Aθ_r', b[:, end])
        p[:, end] = solve(prob)

        for t ∈ (g.nt-1):-1:1
            prob = LinearProblem(Aθ_r', b[:, t] + V_rk' * g.c * g.ϕ * V_rk * p[:, t+1])
            p[:, t] = solve(prob)
        end

        return vec(p)

    end
    
    function compute_∂Ax∂ηtx(
        ∂Ax∂θ::AbstractMatrix, 
        η::AbstractVector, 
        θ::AbstractVector, 
        x::AbstractVector
    )::AbstractVector

        ω_σ, ω_l = η[end-1:end]

        σ = gauss_to_unif(ω_σ, pr.σ_bounds...)
        l = gauss_to_unif(ω_l, pr.l_bounds...)

        α = σ^2 * (4π * gamma(2)) / gamma(1)

        H = pr.M + l^2 * pr.K + l / 1.42 * pr.N

        ∂Ax∂θtx = ∂Ax∂θ' * x
        H∂Ax∂θtx = sparsevec(solve(LinearProblem(H, ∂Ax∂θtx)))

        # White noise component 
        ∂Ax∂ξtx = √(α) * l * pr.L' * H∂Ax∂θtx

        # Standard deviation component
        ∂Ax∂σtx = ((θ .- pr.μ) / σ)' * ∂Ax∂θtx
        ∂Ax∂ξσtx = pr.Δσ * pdf(Normal(), ω_σ) * ∂Ax∂σtx
        
        # Lengthscale component
        ∂Ax∂ltx = (θ - pr.μ)' * (l^-1.0 * pr.M - l * pr.K)' * H∂Ax∂θtx
        ∂Ax∂ξltx = pr.Δl * pdf(Normal(), ω_l) * ∂Ax∂ltx

        return vcat(∂Ax∂ξtx, ∂Ax∂ξσtx, ∂Ax∂ξltx)

    end

    function compute_∂Ax∂ηx(
        ∂Ax∂θ::AbstractMatrix, 
        η::AbstractVector, 
        θ::AbstractVector, 
        x::AbstractVector
    )::AbstractVector

        ω_σ, ω_l = η[end-1:end]

        σ = gauss_to_unif(ω_σ, pr.σ_bounds...)
        l = gauss_to_unif(ω_l, pr.l_bounds...)

        α = σ^2 * (4π * gamma(2)) / gamma(1)

        H = pr.M + l^2 * pr.K + l / 1.42 * pr.N 

        # White noise component
        ∂θ∂ξx = solve(LinearProblem(H, √(α) * l * pr.L * x[1:end-2]))
        ∂Ax∂ξx = ∂Ax∂θ * sparsevec(∂θ∂ξx)

        # Standard deviation component
        ∂σ∂ωσx = pr.Δσ * pdf(Normal(), ω_σ) * x[end-1]
        ∂Ax∂ωσx = ∂Ax∂θ * (sparsevec(θ .- pr.μ) / σ) * ∂σ∂ωσx

        # Lengthscale component
        ∂l∂ωlx = pr.Δl * pdf(Normal(), ω_l) * x[end]
        ∂θ∂ωlx = solve(LinearProblem(H, (l^-1.0 * pr.M - l * pr.K) * (θ .- pr.μ) * ∂l∂ωlx))
        ∂Ax∂ωlx = ∂Ax∂θ * sparsevec(∂θ∂ωlx)

        return ∂Ax∂ξx + ∂Ax∂ωσx + ∂Ax∂ωlx

    end

    function compute_∇Lη(
        ∂Au∂θ::AbstractMatrix,
        ∂Aμ∂θ::AbstractMatrix,
        η::AbstractVector,
        θ::AbstractVector,
        p::AbstractVector
    )::AbstractVector

        return η + compute_∂Ax∂ηtx(∂Au∂θ, η, θ, p) + compute_∂Ax∂ηtx(∂Aμ∂θ, η, θ, p)

    end

    function solve_forward_inc(
        Aθ_r::AbstractMatrix,
        b::AbstractVector
    )::AbstractVector

        b = reshape(b, nu_r, g.nt)
        u = zeros(nu_r, g.nt)

        prob = LinearProblem(Aθ_r, b[:, 1])
        u[:, 1] = solve(prob)

        for t ∈ 2:g.nt 
            prob = LinearProblem(Aθ_r, b[:, t] + g.ϕ * g.c * V_rk' * V_rk * u[:, t-1])
            u[:, t] = solve(prob)
        end

        return vec(u)

    end

    function solve_adjoint_inc(
        Aθ_r::AbstractMatrix,
        u_inc::AbstractVector
    )::AbstractVector
        
        b = BV_r' * Γ_e_inv * BV_r * u_inc
        b = reshape(b, nu_r, g.nt)

        p = zeros(nu_r, g.nt)
        p[:, end] = solve(LinearProblem(Aθ_r', b[:, end]))

        for t ∈ (g.nt-1):-1:1
            prob = LinearProblem(Aθ_r', b[:, t] + V_rk' * g.c * g.ϕ * V_rk * p[:, t+1])
            p[:, t] = solve(prob)
        end

        return vec(p)

    end

    function compute_Hd(
        d::AbstractVector, 
        η::AbstractVector,
        θ::AbstractVector,
        Aθ_r::AbstractMatrix, 
        ∂Au∂θ::AbstractMatrix,
        ∂Aμ∂θ::AbstractMatrix
    )::AbstractVector

        ∂Au∂ηx = compute_∂Ax∂ηx(∂Au∂θ, η, θ, d)
        ∂Aμ∂ηx = compute_∂Ax∂ηx(∂Aμ∂θ, η, θ, d)

        u_inc = solve_forward_inc(Aθ_r, ∂Au∂ηx + ∂Aμ∂ηx)
        p_inc = solve_adjoint_inc(Aθ_r, u_inc)

        ∂Au∂ηtp_inc = compute_∂Ax∂ηtx(∂Au∂θ, η, θ, p_inc)
        ∂Aμ∂ηtp_inc = compute_∂Ax∂ηtx(∂Aμ∂θ, η, θ, p_inc)

        return ∂Au∂ηtp_inc + ∂Aμ∂ηtp_inc + d

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

        i_ls = 1
        while i_ls < LS_MAX_ITS

            η_k = η_c + α_k * δη
            θ_k = transform(pr, η_k)

            Aθ = g.c * g.ϕ * sparse(I, g.nx^2, g.nx^2) + (g.Δt / g.μ) * g.∇h' * spdiagm(g.A * exp.(θ_k)) * g.∇h
            Aθ_r = V_rk' * Aθ * V_rk 

            # TODO: return this to the main loop to avoid computing it again at the next iteration
            u_k = solve_forward(Aθ, Aθ_r)
            
            J_k = J(η_k, u_k)

            @printf "%6i | %.3e\n" i_ls J_k 

            if (J_k ≤ J_c + LS_C * α_k * ∇Lη_c' * δη)
                println("Linesearch converged after $i_ls iterations.")
                return α_k
            end

            α_k *= 0.5
            i_ls += 1

        end

        @warn "Linesearch failed to converge within $LS_MAX_ITS iterations."
        return α_k

    end

    norm∇Lη0 = nothing
    i_gn = 1
    while true

        @info "Beginning GN It. $i_gn"
        
        θ = transform(pr, η)

        Aθ = g.c * g.ϕ * sparse(I, g.nx^2, g.nx^2) + (g.Δt / g.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h
        Aθ_r = V_rk' * Aθ * V_rk

        u = solve_forward(Aθ, Aθ_r)
        p = solve_adjoint(u, Aθ_r)

        ∂Au∂θ = compute_∂Au∂θ(θ, u)
        ∂Aμ∂θ = compute_∂Aμ∂θ(θ)

        ∇Lη = compute_∇Lη(∂Au∂θ, ∂Aμ∂θ, η, θ, p)
        
        if i_gn == 1
            norm∇Lη0 = norm(∇Lη)
        end
        tol_cg = min(0.5, √(norm(∇Lη) / norm∇Lη0)) * norm(∇Lη)

        @printf "norm(∇Lη): %.3e\n" norm(∇Lη)
        @printf "CG tolerance: %3.e\n" tol_cg

        if norm(∇Lη) < GN_MIN_NORM
            return η, u
        elseif i_gn > GN_MAX_ITS
            @warn "Gauss-Newton failed to converge within $GN_MAX_ITS iterations."
            return η, u
        end

        println("CG It. | norm(r)")
        δη = spzeros(pr.Nη)
        d = -copy(∇Lη)
        r = -copy(∇Lη)

        i_cg = 1
        while true

            Hd = compute_Hd(d, η, θ, Aθ_r, ∂Au∂θ, ∂Aμ∂θ)
 
            α = (r' * r) / (d' * Hd)
            δη += α * d

            r_prev = copy(r)
            r = r_prev - α * Hd

            @printf "%6i | %.3e\n" i_cg norm(r)

            if (norm(r) ≤ tol_cg)
                println("CG converged after $i_cg iterations.")
                break
            elseif i_cg > CG_MAX_ITS
                @warn "CG failed to converge within $CG_MAX_ITS iterations."
            end
            
            β = (r' * r) / (r_prev' * r_prev)
            d = r + β * d

            i_cg += 1

        end

        α = linesearch(η, u, ∇Lη, δη)
        η += α * δη
        i_gn += 1

    end

end