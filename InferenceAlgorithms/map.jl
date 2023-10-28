using Distributions
using KrylovKit
using LinearAlgebra
using LinearSolve
using Printf
using SparseArrays

# TODO: count number of PDE solves somewhere

const GN_MIN_NORM = 1e-2
const GN_MAX_ITS = 30
const CG_MAX_ITS = 30
const LS_C = 1e-4
const LS_MAX_ITS = 20

const UNIT_NORM = Normal()

struct GNResult

    converged::Bool
    η::AbstractVector
    θ::AbstractVector
    u_r::AbstractVector

    Aθ_r::AbstractMatrix
    ∂Au∂θ::AbstractMatrix
    ∂Aμ∂θ::AbstractMatrix

end

function get_full_state(
    u_r::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::AbstractVector
    
    u_r = reshape(u_r, m.nu_r, g.nt)
    u = vec(m.V_ri * u_r .+ m.μ_ui)    
    return u

end

function J(
    η::AbstractVector, 
    u_r::AbstractVector,
    d_obs::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::Real

    u = get_full_state(u_r, g, m)
    res = m.B * u + m.μ_e - d_obs
    return 0.5 * res' * m.Γ_e_inv * res + 0.5 * sum(η.^2)

end

function compute_∂Au∂θ(
    θ::AbstractVector, 
    u_r::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::AbstractMatrix

    u_r = reshape(u_r, m.nu_r, g.nt)

    ∂Au∂θ = (g.Δt / m.μ) * vcat([
        m.V_ri' * g.∇h' * spdiagm(g.∇h * m.V_ri * u_r[:, t]) * 
        g.A * spdiagm(exp.(θ)) for t ∈ 1:g.nt
    ]...)

    return ∂Au∂θ

end

function compute_∂Aμ∂θ(
    θ::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::AbstractMatrix 

    ∂Aμ∂θ = (g.Δt / m.μ) * vcat([
        m.V_ri' * g.∇h' * spdiagm(g.∇h * m.μ_ui) * 
        g.A * spdiagm(exp.(θ)) for t ∈ 1:g.nt
    ]...)

    return ∂Aμ∂θ

end

function solve_forward(
    Aθ::AbstractMatrix,
    Aθ_r::AbstractMatrix,
    g::Grid,
    m::ReducedOrderModel
)::AbstractVector

    u_r = zeros(m.nu_r, g.nt)

    b = m.V_ri' * (g.Δt * m.Q[:, 1] .+ (m.c * m.ϕ * m.u0) .- Aθ * m.μ_ui)
    u_r[:, 1] = solve(LinearProblem(Aθ_r, b)).u

    for t ∈ 2:g.nt 
        bt = m.V_ri' * (g.Δt * m.Q[:, t] - Aθ * m.μ_ui) + m.ϕ * m.c * (u_r[:, t-1] + m.V_ri' * m.μ_ui)
        u_r[:, t] = solve(LinearProblem(Aθ_r, bt)).u
    end

    return vec(u_r)

end

function solve_adjoint(
    u_r::AbstractVector, 
    Aθ_r::AbstractMatrix,
    d_obs::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::AbstractVector 

    u = get_full_state(u_r, g, m)
    p = zeros(m.nu_r, g.nt)

    b = -m.BV_r' * m.Γ_e_inv * (m.B * u + m.μ_e - d_obs)
    b = reshape(b, m.nu_r, g.nt)

    p[:, end] = solve(LinearProblem(Matrix(Aθ_r'), b[:, end])).u

    for t ∈ (g.nt-1):-1:1
        bt = b[:, t] + m.c * m.ϕ * p[:, t+1]
        p[:, t] = solve(LinearProblem(Matrix(Aθ_r'), bt)).u
    end

    return vec(p)

end

function compute_∂Ax∂ηtx(
    ∂Ax∂θ::AbstractMatrix, 
    η::AbstractVector, 
    θ::AbstractVector, 
    x::AbstractVector,
    pr::MaternField
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
    ∂Ax∂ωσtx = pr.Δσ * pdf(UNIT_NORM, ω_σ) * ∂Ax∂σtx
    
    # Lengthscale component
    ∂Ax∂ltx = (θ - pr.μ)' * (l^-1.0 * pr.M - l * pr.K) * H∂Ax∂θtx
    ∂Ax∂ωltx = pr.Δl * pdf(UNIT_NORM, ω_l) * ∂Ax∂ltx

    return vcat(∂Ax∂ξtx, ∂Ax∂ωσtx, ∂Ax∂ωltx)

end

function compute_∂Ax∂ηx(
    ∂Ax∂θ::AbstractMatrix, 
    η::AbstractVector, 
    θ::AbstractVector, 
    x::AbstractVector,
    pr::MaternField
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
    ∂σ∂ωσx = pr.Δσ * pdf(UNIT_NORM, ω_σ) * x[end-1]
    ∂Ax∂ωσx = ∂Ax∂θ * (θ - pr.μ) / σ * ∂σ∂ωσx

    # Lengthscale component
    ∂l∂ωlx = pr.Δl * pdf(UNIT_NORM, ω_l) * x[end]
    ∂θ∂ωlx = solve(LinearProblem(H, (l^-1.0 * pr.M - l * pr.K) * (θ .- pr.μ) * ∂l∂ωlx)).u
    ∂Ax∂ωlx = ∂Ax∂θ * ∂θ∂ωlx

    return ∂Ax∂ξx + ∂Ax∂ωσx + ∂Ax∂ωlx

end

function compute_∇Lη(
    ∂Au∂θ::AbstractMatrix,
    ∂Aμ∂θ::AbstractMatrix,
    η::AbstractVector,
    θ::AbstractVector,
    p::AbstractVector,
    pr::MaternField
)::AbstractVector

    return η + 
        compute_∂Ax∂ηtx(∂Au∂θ, η, θ, p, pr) + 
        compute_∂Ax∂ηtx(∂Aμ∂θ, η, θ, p, pr)

end

function solve_forward_inc(
    Aθ_r::AbstractMatrix,
    b::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::AbstractVector

    b = reshape(b, m.nu_r, g.nt)
    u = zeros(m.nu_r, g.nt)

    u[:, 1] = solve(LinearProblem(Aθ_r, b[:, 1])).u

    for t ∈ 2:g.nt 
        bt = b[:, t] + m.ϕ * m.c * u[:, t-1]
        u[:, t] = solve(LinearProblem(Aθ_r, bt)).u
    end

    return vec(u)

end

function solve_adjoint_inc(
    Aθ_r::AbstractMatrix,
    u_inc::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::AbstractVector
    
    Aθ_rt = Matrix(Aθ_r')

    b = m.BV_r' * m.Γ_e_inv * m.BV_r * u_inc
    b = reshape(b, m.nu_r, g.nt)

    p = zeros(m.nu_r, g.nt)
    p[:, end] = solve(LinearProblem(Aθ_rt, b[:, end])).u

    for t ∈ (g.nt-1):-1:1
        bt = b[:, t] + m.c * m.ϕ * p[:, t+1]
        p[:, t] = solve(LinearProblem(Aθ_rt, bt)).u
    end

    return vec(p)

end

function compute_Hx(
    x::AbstractVector, 
    η::AbstractVector,
    θ::AbstractVector,
    Aθ_r::AbstractMatrix, 
    ∂Au∂θ::AbstractMatrix,
    ∂Aμ∂θ::AbstractMatrix,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField
)::AbstractVector

    ∂Au∂ηx = compute_∂Ax∂ηx(∂Au∂θ, η, θ, x, pr)
    ∂Aμ∂ηx = compute_∂Ax∂ηx(∂Aμ∂θ, η, θ, x, pr)

    u_inc = solve_forward_inc(Aθ_r, ∂Au∂ηx + ∂Aμ∂ηx, g, m)
    p_inc = solve_adjoint_inc(Aθ_r, u_inc, g, m)

    ∂Au∂ηtp_inc = compute_∂Ax∂ηtx(∂Au∂θ, η, θ, p_inc, pr)
    ∂Aμ∂ηtp_inc = compute_∂Ax∂ηtx(∂Aμ∂θ, η, θ, p_inc, pr)

    return ∂Au∂ηtp_inc + ∂Aμ∂ηtp_inc + x

end

function linesearch(
    η_c::AbstractVector, 
    u_r_c::AbstractVector, 
    ∇Lη_c::AbstractVector, 
    δη::AbstractVector,
    d_obs::AbstractVector,
    g::Grid, 
    m::ReducedOrderModel,
    pr::MaternField
)::Tuple{AbstractVector, AbstractVector}

    println("LS It. | J(η, u)")
    J_c = J(η_c, u_r_c, d_obs, g, m)
    
    α_k = 1.0
    η_k = nothing
    u_r_k = nothing

    i_ls = 1
    while i_ls < LS_MAX_ITS

        η_k = η_c + α_k * δη
        θ_k = transform(pr, η_k)

        Aθ = m.c * m.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(θ_k)) * g.∇h
        Aθ_r = m.V_ri' * Aθ * m.V_ri 

        u_r_k = solve_forward(Aθ, Aθ_r, g, m)
        J_k = J(η_k, u_r_k, d_obs, g, m)

        @printf "%6i | %.3e\n" i_ls J_k 

        if (J_k ≤ J_c + LS_C * α_k * ∇Lη_c' * δη)
            println("Linesearch converged after $i_ls iterations.")
            return η_k, u_r_k
        end

        α_k *= 0.5
        i_ls += 1

    end

    @warn "Linesearch failed to converge within $LS_MAX_ITS iterations."
    return η_k, u_r_k

end

function solve_cg(
    η::AbstractVector,
    θ::AbstractVector,
    Aθ_r::AbstractMatrix,
    ∂Au∂θ::AbstractMatrix,
    ∂Aμ∂θ::AbstractMatrix,
    ∇Lη::AbstractVector,
    tol::Real,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField
)::AbstractVector

    println("CG It. | norm(r)")
    δη = spzeros(pr.Nη)
    d = -copy(∇Lη)
    r = -copy(∇Lη)

    i = 1
    while true

        Hd = compute_Hx(d, η, θ, Aθ_r, ∂Au∂θ, ∂Aμ∂θ, g, m, pr)

        α = (r' * r) / (d' * Hd)
        δη += α * d

        r_prev = copy(r)
        r = r_prev - α * Hd

        @printf "%6i | %.3e\n" i norm(r)

        if norm(r) < tol
            println("CG converged after $i iterations.")
            return δη
        elseif i > CG_MAX_ITS
            @warn "CG failed to converge within $CG_MAX_ITS iterations."
            return δη
        end
        
        β = (r' * r) / (r_prev' * r_prev)
        d = r + β * d
        i += 1

    end

end

function compute_map(
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField,
    d_obs::AbstractVector,
    η0::AbstractVector,
)

    η = copy(η0) 
    u_r = nothing
    norm∇Lη0 = nothing
    
    i = 1
    while true

        @info "Beginning GN It. $i"
        
        θ = transform(pr, η)

        Aθ = m.c * m.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h
        Aθ_r = m.V_ri' * Aθ * m.V_ri

        if i == 1
            u_r = solve_forward(Aθ, Aθ_r, g, m)
        end
        p = solve_adjoint(u_r, Aθ_r, d_obs, g, m)

        # TODO: how to make these faster?
        ∂Au∂θ = compute_∂Au∂θ(θ, u_r, g, m)
        ∂Aμ∂θ = compute_∂Aμ∂θ(θ, g, m)

        ∇Lη = compute_∇Lη(∂Au∂θ, ∂Aμ∂θ, η, θ, p, pr)
        if norm(∇Lη) < GN_MIN_NORM
            println("Converged.")
            return GNResult(true, η, θ, u_r, Aθ_r, ∂Au∂θ, ∂Aμ∂θ)
        end
        
        if i == 1
            norm∇Lη0 = norm(∇Lη)
        end
        tol_cg = 0.1 * min(0.5, √(norm(∇Lη) / norm∇Lη0)) * norm(∇Lη)

        @printf "norm(∇Lη): %.3e\n" norm(∇Lη)
        @printf "CG tolerance: %.3e\n" tol_cg

        δη = solve_cg(η, θ, Aθ_r, ∂Au∂θ, ∂Aμ∂θ, ∇Lη, tol_cg, g, m, pr)
        η, u_r = linesearch(η, u_r, ∇Lη, δη, d_obs, g, m, pr)
        
        i += 1
        if i > GN_MAX_ITS
            @warn "Failed to converge within $GN_MAX_ITS iterations."
            return GNResult(false, η, θ, u_r, Aθ_r, ∂Au∂θ, ∂Aμ∂θ)
        end

    end

end

function compute_laplace(
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField,
    d_obs::AbstractVector,
    η0::AbstractVector;
    n_eigvals::Int=500
)

    map = compute_map(g, m, pr, d_obs, η0)
    !map.converged && @warn "MAP optimisation failed to converge."

    f(x) = compute_Hx(x, map.η, map.θ, map.Aθ_r, map.∂Au∂θ, map.∂Aμ∂θ, g, m, pr)

    vals, vecs, info = eigsolve(f, pr.Nη, n_eigvals, :LR, krylovdim=10000, issymmetric=true, tol=1e-16)
    println(vals)
    println(info.converged)
    println(info.numops)
    println(info.numiter)

    return -1

end