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
    μ_u::AbstractVector,        # Mean of u, estimated using samples
    V_r::AbstractMatrix,        # Reduced basis for u
    μ_ε::AbstractVector,        # Mean of model errors 
    Γ_e_inv::AbstractMatrix     # Inverse of combined measurement and model error covariance
)

    # TODO: move to MaternField struct?
    Δσ = pr.σ_bounds[2] - pr.σ_bounds[1]
    Δl = pr.l_bounds[2] - pr.l_bounds[1]

    # Get the size of the reduced state vector
    nu_r = size(V_r, 2)

    V_r_f = sparse(kron(sparse(I, g.nt, g.nt), V_r))
    BV_r = g.B * V_r_f
    μ_u_f = repeat(μ_u, g.nt)

    function J(
        η::AbstractVector, 
        u::AbstractVector
    )::Real
        res = g.B * (V_r_f * u + μ_u_f) + μ_ε - y
        return 0.5 * res' * Γ_e_inv * res + 0.5 * sum(η.^2)
    end

    function compute_∂Au∂θ(
        θ::AbstractVector, 
        u::AbstractVector
    )::AbstractMatrix

        u = reshape(u, nu_r, g.nt)

        ∂Au∂θ = (g.Δt / g.μ) * vcat([
            V_r' * g.∇h' * spdiagm(g.∇h * V_r * u[:, t]) * 
            g.A * spdiagm(exp.(θ)) for t ∈ 1:g.nt
        ]...)

        return ∂Au∂θ

    end

    function compute_∂Aμ∂θ(
        θ::AbstractVector
    )::AbstractMatrix 

        ∂Aμ∂θ = (g.Δt / g.μ) * vcat([
            V_r' * g.∇h' * spdiagm(g.∇h * μ_u) * 
            g.A * spdiagm(exp.(θ)) for t ∈ 1:g.nt
        ]...)

        return ∂Aμ∂θ

    end

    function solve_forward(
        Bθ::AbstractMatrix,
        B̃θ::AbstractMatrix
    )::AbstractVector

        u = zeros(nu_r, g.nt)

        b = V_r' * (g.Δt * Q[:, 1] .+ (g.ϕ * g.c * u0) .- Bθ * μ_u)
        u[:, 1] = solve(LinearProblem(B̃θ, b))

        for t ∈ 2:g.nt 
            b = V_r' * (g.Δt * Q[:, t] + g.ϕ * g.c * (V_r * u[:, t-1] + μ_u) - Bθ * μ_u)
            u[:, t] = solve(LinearProblem(B̃θ, b))
        end

        return vec(u)

    end

    function solve_adjoint(
        u::AbstractVector, 
        B̃θ::AbstractMatrix
    )::AbstractVector
        
        p = zeros(nu_r, g.nt) 

        b = -BV_r' * Γ_e_inv * (BV_r * u + g.B * μ_u_f + μ_ε - y) 
        b = reshape(b, nu_r, g.nt)

        prob = LinearProblem(B̃θ', b[:, end])
        p[:, end] = solve(prob)

        for t ∈ (g.nt-1):-1:1
            prob = LinearProblem(B̃θ', b[:, t] + g.ϕ * g.c * V_r' * V_r * p[:, t+1])
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

        ξ_σ, ξ_l = η[end-1:end]

        σ = gauss_to_unif(ξ_σ, pr.σ_bounds...)
        l = gauss_to_unif(ξ_l, pr.l_bounds...)

        α = σ^2 * (4π * gamma(2)) / gamma(1)

        H = pr.M + l^2 * pr.K + l / 1.42 * pr.N

        ∂Ax∂θtx = ∂Ax∂θ' * x
        H∂Ax∂θtx = sparsevec(solve(LinearProblem(H, ∂Ax∂θtx)))

        # White noise component 
        ∂Ax∂ξtx = √(α) * l * pr.L' * H∂Ax∂θtx

        # Standard deviation component
        ∂Ax∂σtx = ((θ .- pr.μ) / σ)' * ∂Ax∂θtx
        ∂Ax∂ξσtx = Δσ * pdf(Normal(), ξ_σ) * ∂Ax∂σtx
        
        # Lengthscale component
        ∂Ax∂ltx = -(θ - pr.μ)' * (-l^-1.0 * pr.M + l * pr.K)' * H∂Ax∂θtx
        ∂Ax∂ξltx = Δl * pdf(Normal(), ξ_l) * ∂Ax∂ltx

        return vcat(∂Ax∂ξtx, ∂Ax∂ξσtx, ∂Ax∂ξltx)

    end

    function compute_∂Ax∂ηx(
        ∂Ax∂θ::AbstractMatrix, 
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
        ∂Ax∂ξx = ∂Ax∂θ * sparsevec(∂θ∂ξx)

        # Standard deviation component
        ∂σ∂ξσx = Δσ * pdf(Normal(), ξ_σ) * x[end-1]
        ∂Ax∂ξσx = ∂Ax∂θ * (sparsevec(θ .- pr.μ) / σ) * ∂σ∂ξσx

        # Lengthscale component
        ∂l∂ξlx = Δl * pdf(Normal(), ξ_l) * x[end]
        ∂θ∂ξlx = -solve(LinearProblem(H, (-l^-1.0 * pr.M + l * pr.K) * (θ .- pr.μ) * ∂l∂ξlx))
        ∂Ax∂ξlx = ∂Ax∂θ * sparsevec(∂θ∂ξlx)

        return ∂Ax∂ξx + ∂Ax∂ξσx + ∂Ax∂ξlx

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
        B̃θ::AbstractMatrix,
        b::AbstractVector
    )::AbstractVector

        b = reshape(b, nu_r, g.nt)
        u = zeros(nu_r, g.nt)

        prob = LinearProblem(B̃θ, b[:, 1])
        u[:, 1] = solve(prob)

        for t ∈ 2:g.nt 
            prob = LinearProblem(B̃θ, b[:, t] + g.ϕ * g.c * V_r' * V_r * u[:, t-1])
            u[:, t] = solve(prob)
        end

        return vec(u)

    end

    function solve_adjoint_inc(
        B̃θ::AbstractMatrix,
        u_inc::AbstractVector
    )::AbstractVector
        
        b = BV_r' * Γ_e_inv * BV_r * u_inc
        b = reshape(b, nu_r, g.nt)

        p = zeros(nu_r, g.nt)
        p[:, end] = solve(LinearProblem(B̃θ', b[:, end]))

        for t ∈ (g.nt-1):-1:1
            prob = LinearProblem(B̃θ', b[:, t] + g.ϕ * g.c * V_r' * V_r * p[:, t+1])
            p[:, t] = solve(prob)
        end

        return vec(p)

    end

    function compute_Hd(
        d::AbstractVector, 
        η::AbstractVector,
        θ::AbstractVector,
        B̃θ::AbstractMatrix, 
        ∂Au∂θ::AbstractMatrix,
        ∂Aμ∂θ::AbstractMatrix
    )::AbstractVector

        ∂Au∂ηx = compute_∂Ax∂ηx(∂Au∂θ, η, θ, d)
        ∂Aμ∂ηx = compute_∂Ax∂ηx(∂Aμ∂θ, η, θ, d)

        u_inc = solve_forward_inc(B̃θ, ∂Au∂ηx + ∂Aμ∂ηx)
        p_inc = solve_adjoint_inc(B̃θ, u_inc)

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
            
            Aθ_k = (1.0 / g.μ) * g.∇h' * spdiagm(g.A * exp.(θ_k)) * g.∇h
            Bθ_k = g.ϕ * g.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * Aθ_k
            B̃θ_k = V_r' * Bθ_k * V_r 

            # TODO: return this to the main loop to avoid computing it again at the next iteration
            u_k = solve_forward(Bθ_k, B̃θ_k)
            
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

        Aθ = (1.0 / g.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h
        Bθ = g.ϕ * g.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * Aθ
        B̃θ = V_r' * Bθ * V_r

        u = solve_forward(Bθ, B̃θ)
        p = solve_adjoint(u, B̃θ)

        ∂Au∂θ = compute_∂Au∂θ(θ, u)
        ∂Aμ∂θ = compute_∂Aμ∂θ(θ)

        ∇Lη = @time compute_∇Lη(∂Au∂θ, ∂Aμ∂θ, η, θ, p)
        
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
        δη = spzeros(pr.Nθ)
        d = -copy(∇Lη)
        r = -copy(∇Lη)

        i_cg = 1
        while true

            Hd = compute_Hd(d, η, θ, B̃θ, ∂Au∂θ, ∂Aμ∂θ)
 
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

# TODO: remove
# η = vec(rand(pr, 1))
# η_map, u_map = optimise(grid_c, pr, y_obs, Q_c, η, μ_u, V_r, μ_ε, Γ_e_inv)