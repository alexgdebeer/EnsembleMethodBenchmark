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
    μ_ui::AbstractVector,       # Mean of u, estimated using samples
    V_ri::AbstractMatrix,       # Reduced basis for u
    μ_e::AbstractVector,        # Mean of errors 
    Γ_e_inv::AbstractMatrix     # Inverse of combined measurement and model error covariance
)

    unit_norm = Normal()

    # Get the size of the reduced state vector
    nu_r = size(V_ri, 2)

    # TODO: avoid using these variables, or include them as inputs
    V_r = sparse(kron(sparse(I, g.nt, g.nt), V_ri))
    BV_r = g.B * V_r

    function J(
        η::AbstractVector, 
        u_r::AbstractVector
    )::Real

        # TODO: make helper for this?
        u_r = reshape(u_r, nu_r, g.nt)
        u = vec(V_ri * u_r .+ μ_ui)

        res = g.B * u + μ_e - y
        return 0.5 * res' * Γ_e_inv * res + 0.5 * sum(η.^2)

    end

    function compute_∂Au∂θ(
        θ::AbstractVector, 
        u_r::AbstractVector
    )::AbstractMatrix

        u_r = reshape(u_r, nu_r, g.nt)

        ∂Au∂θ = (g.Δt / g.μ) * vcat([
            V_ri' * g.∇h' * spdiagm(g.∇h * V_ri * u_r[:, t]) * 
            g.A * spdiagm(exp.(θ)) for t ∈ 1:g.nt
        ]...)

        return ∂Au∂θ

    end

    function compute_∂Aμ∂θ(
        θ::AbstractVector
    )::AbstractMatrix 

        ∂Aμ∂θ = (g.Δt / g.μ) * vcat([
            V_ri' * g.∇h' * spdiagm(g.∇h * μ_ui) * 
            g.A * spdiagm(exp.(θ)) for t ∈ 1:g.nt
        ]...)

        return ∂Aμ∂θ

    end

    function solve_forward(
        Aθ::AbstractMatrix,
        Aθ_r::AbstractMatrix
    )::AbstractVector

        u_r = zeros(nu_r, g.nt)

        b = V_ri' * (g.Δt * Q[:, 1] .+ (g.c * g.ϕ * u0) .- Aθ * μ_ui)
        u_r[:, 1] = solve(LinearProblem(Aθ_r, b)).u

        for t ∈ 2:g.nt 
            b = V_ri' * (g.Δt * Q[:, t] + g.ϕ * g.c * (V_ri * u_r[:, t-1] + μ_ui) - Aθ * μ_ui)
            u_r[:, t] = solve(LinearProblem(Aθ_r, b)).u
        end

        return vec(u_r)

    end

    function solve_adjoint(
        u_r::AbstractVector, 
        Aθ_r::AbstractMatrix
    )::AbstractVector
        
        p = zeros(nu_r, g.nt) 

        # TODO: make helper for this?
        u_r = reshape(u_r, nu_r, g.nt)
        u = vec(V_ri * u_r .+ μ_ui)

        b = -V_r' * g.B' * Γ_e_inv * (g.B * u + μ_e - y) # TODO: how to get rid of first V_r?
        b = reshape(b, nu_r, g.nt)

        p[:, end] = solve(LinearProblem(Matrix(Aθ_r'), b[:, end])).u

        for t ∈ (g.nt-1):-1:1
            bt = b[:, t] + V_ri' * g.c * g.ϕ * V_ri * p[:, t+1]
            p[:, t] = solve(LinearProblem(Matrix(Aθ_r'), bt)).u
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
        H∂Ax∂θtx = solve(LinearProblem(H, ∂Ax∂θtx)).u

        # White noise component
        ∂Ax∂ξtx = √(α) * l * pr.L' * H∂Ax∂θtx

        # Standard deviation component
        ∂Ax∂σtx = ((θ .- pr.μ) / σ)' * ∂Ax∂θtx
        ∂Ax∂ωσtx = pr.Δσ * pdf(unit_norm, ω_σ) * ∂Ax∂σtx
        
        # Lengthscale component
        ∂Ax∂ltx = (θ - pr.μ)' * (l^-1.0 * pr.M - l * pr.K) * H∂Ax∂θtx
        ∂Ax∂ωltx = pr.Δl * pdf(unit_norm, ω_l) * ∂Ax∂ltx

        return vcat(∂Ax∂ξtx, ∂Ax∂ωσtx, ∂Ax∂ωltx)

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
        ∂θ∂ξx = solve(LinearProblem(H, √(α) * l * pr.L * x[1:end-2])).u
        ∂Ax∂ξx = ∂Ax∂θ * ∂θ∂ξx

        # Standard deviation component
        ∂σ∂ωσx = pr.Δσ * pdf(unit_norm, ω_σ) * x[end-1]
        ∂Ax∂ωσx = ∂Ax∂θ * (θ - pr.μ) / σ * ∂σ∂ωσx

        # Lengthscale component
        ∂l∂ωlx = pr.Δl * pdf(unit_norm, ω_l) * x[end]
        ∂θ∂ωlx = solve(LinearProblem(H, (l^-1.0 * pr.M - l * pr.K) * (θ .- pr.μ) * ∂l∂ωlx)).u
        ∂Ax∂ωlx = ∂Ax∂θ * ∂θ∂ωlx

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

        u[:, 1] = solve(LinearProblem(Aθ_r, b[:, 1])).u

        for t ∈ 2:g.nt 
            bt = b[:, t] + g.ϕ * g.c * V_ri' * V_ri * u[:, t-1]
            u[:, t] = solve(LinearProblem(Aθ_r, bt)).u
        end

        return vec(u)

    end

    function solve_adjoint_inc(
        Aθ_r::AbstractMatrix,
        u_inc::AbstractVector
    )::AbstractVector
        
        Aθ_rt = Matrix(Aθ_r')

        b = BV_r' * Γ_e_inv * BV_r * u_inc
        b = reshape(b, nu_r, g.nt)

        p = zeros(nu_r, g.nt)
        p[:, end] = solve(LinearProblem(Aθ_rt, b[:, end])).u

        for t ∈ (g.nt-1):-1:1
            bt = b[:, t] + V_ri' * g.c * g.ϕ * V_ri * p[:, t+1]
            p[:, t] = solve(LinearProblem(Aθ_rt, bt)).u
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
        u_r_c::AbstractVector, 
        ∇Lη_c::AbstractVector, 
        δη::AbstractVector
    )::Real

        println("LS It. | J(η, u)")
        J_c = J(η_c, u_r_c)
        α_k = 1.0

        i_ls = 1
        while i_ls < LS_MAX_ITS

            η_k = η_c + α_k * δη
            θ_k = transform(pr, η_k)

            Aθ = g.c * g.ϕ * sparse(I, g.nx^2, g.nx^2) + 
                (g.Δt / g.μ) * g.∇h' * spdiagm(g.A * exp.(θ_k)) * g.∇h
            Aθ_r = V_ri' * Aθ * V_ri 

            # TODO: return this to the main loop to avoid computing it again at the next iteration
            u_r_k = solve_forward(Aθ, Aθ_r)
            
            J_k = J(η_k, u_r_k)

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

        Aθ = g.c * g.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / g.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h
        Aθ_r = V_ri' * Aθ * V_ri

        u_r = solve_forward(Aθ, Aθ_r)
        p = solve_adjoint(u_r, Aθ_r)

        # println(J(η, u))

        # # ----------------
        # # TEMP: test forward problem by solving full problem.
        # # ----------------
        # Ãθ = blockdiag([Aθ for _ ∈ 1:g.nt]...)
        # Id = g.c * g.ϕ * sparse(I, g.nx^2*(g.nt-1), g.nx^2*(g.nt-1))
        # Ãθ[(g.nx^2+1):end, 1:(g.nx^2*(g.nt-1))] -= Id
        # B̃θ = blockdiag([Aθ for _ ∈ 1:g.nt]...)
        # Qs = sparsevec(Q)

        # û0 = spzeros(g.nx^2*g.nt)
        # û0[1:g.nx^2] = g.u0 .- μ_ui
        # u_test = solve(LinearProblem(Matrix(V_r' * Ãθ * V_r), V_r' * (g.Δt * Qs + g.c * g.ϕ * (û0 + μ_u) - B̃θ * μ_u)))

        # # ----------------
        # # TEMP: test adjoint problem by solving full problem.
        # # ----------------
        # b_test = -V_r' * g.B' * Γ_e_inv * (g.B * (V_r * u + μ_u) + μ_e - y)
        # p_test = solve(LinearProblem(Matrix(V_r' * Ãθ' * V_r), Vector(b_test))).u

        # p = reshape(p, nu_r, g.nt)
        # p_test = reshape(p_test, nu_r, g.nt)
        # display(p)
        # display(p_test)
        # error("stop")

        # TODO: how to make these faster?
        ∂Au∂θ = compute_∂Au∂θ(θ, u_r)
        ∂Aμ∂θ = compute_∂Aμ∂θ(θ)

        ∇Lη = compute_∇Lη(∂Au∂θ, ∂Aμ∂θ, η, θ, p)
        
        if i_gn == 1
            norm∇Lη0 = norm(∇Lη)
        end
        tol_cg = min(0.5, √(norm(∇Lη) / norm∇Lη0)) * norm(∇Lη)

        @printf "norm(∇Lη): %.3e\n" norm(∇Lη)
        @printf "CG tolerance: %3.e\n" tol_cg

        if norm(∇Lη) < GN_MIN_NORM
            return η, u_r
        elseif i_gn > GN_MAX_ITS
            @warn "Gauss-Newton failed to converge within $GN_MAX_ITS iterations."
            return η, u_r
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

            if norm(r) < tol_cg
                println("CG converged after $i_cg iterations.")
                break
            elseif i_cg > CG_MAX_ITS
                @warn "CG failed to converge within $CG_MAX_ITS iterations."
            end
            
            β = (r' * r) / (r_prev' * r_prev)
            d = r + β * d

            i_cg += 1

        end

        α = linesearch(η, u_r, ∇Lη, δη)
        η += α * δη
        i_gn += 1

    end

end