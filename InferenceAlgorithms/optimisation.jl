const GN_MIN_NORM = 1e-2
const GN_MAX_ITS = 30
const CG_MAX_ITS = 30
const LS_C = 1e-4
const LS_MAX_ITS = 20

const UNIT_NORM = Normal()

struct GNResult

    converged::Bool
    θ::AbstractVector
    u::AbstractVector
    p_r::AbstractVector

    Au_r::AbstractMatrix
    ∂Ap∂u::AbstractMatrix
    ∂Aμ∂u::AbstractMatrix

end

sqnorm(x) = sum(x.^2)

function get_full_state(
    p_r::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::AbstractVector
    
    p_r = reshape(p_r, m.np_r, g.nt)
    p = vec(m.V_ri * p_r .+ m.μ_pi)
    return p

end

function J(
    θ::AbstractVector, 
    p_r::AbstractVector,
    y::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::Real

    p = get_full_state(p_r, g, m)
    res = m.B * p + m.μ_e - y
    return 0.5 * res' * m.C_e_inv * res + 0.5 * sum(θ.^2)

end

function compute_∂Ap∂u(
    u::AbstractVector, 
    p_r::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::AbstractMatrix

    p_r = reshape(p_r, m.np_r, g.nt)

    ∂Au∂u = (g.Δt / m.μ) * vcat([
        m.V_ri' * g.∇h' * spdiagm(g.∇h * m.V_ri * p_r[:, t]) * 
        g.A * spdiagm(exp.(u)) for t ∈ 1:g.nt
    ]...)

    return ∂Au∂u

end

function compute_∂Aμ∂u(
    u::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::AbstractMatrix 

    ∂Aμ∂u = (g.Δt / m.μ) * vcat([
        m.V_ri' * g.∇h' * spdiagm(g.∇h * m.μ_pi) * 
        g.A * spdiagm(exp.(u)) for t ∈ 1:g.nt
    ]...)

    return ∂Aμ∂u

end

function solve_forward(
    Au::AbstractMatrix,
    Au_r::AbstractMatrix,
    g::Grid,
    m::ReducedOrderModel
)::AbstractVector

    p_r = zeros(m.np_r, g.nt)

    b = m.V_ri' * (g.Δt * m.Q[:, 1] .+ (m.c * m.ϕ * m.p0) .- Au * m.μ_pi)
    p_r[:, 1] = solve(LinearProblem(Au_r, b)).u

    for t ∈ 2:g.nt 
        bt = m.V_ri' * (g.Δt * m.Q[:, t] - Au * m.μ_pi) + m.ϕ * m.c * (p_r[:, t-1] + m.V_ri' * m.μ_pi)
        p_r[:, t] = solve(LinearProblem(Au_r, bt)).u
    end

    return vec(p_r)

end

function solve_adjoint(
    p_r::AbstractVector, 
    Au_r::AbstractMatrix,
    y::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::AbstractVector 

    p = get_full_state(p_r, g, m)
    λ = zeros(m.np_r, g.nt)

    b = -m.BV_r' * m.C_e_inv * (m.B * p + m.μ_e - y)
    b = reshape(b, m.np_r, g.nt)

    λ[:, end] = solve(LinearProblem(Matrix(Au_r'), b[:, end])).u

    for t ∈ (g.nt-1):-1:1
        bt = b[:, t] + m.c * m.ϕ * λ[:, t+1]
        λ[:, t] = solve(LinearProblem(Matrix(Au_r'), bt)).u
    end

    return vec(λ)

end

function compute_∂Ax∂θtx(
    ∂Ax∂u::AbstractMatrix, 
    θ::AbstractVector, 
    u::AbstractVector, 
    x::AbstractVector,
    pr::MaternField
)::AbstractVector

    ω_σ, ω_l = θ[end-1:end]

    σ = gauss_to_unif(ω_σ, pr.σ_bounds...)
    l = gauss_to_unif(ω_l, pr.l_bounds...)

    α = σ^2 * (4π * gamma(2)) / gamma(1)

    H = pr.M + l^2 * pr.K + l / 1.42 * pr.N

    ∂Ax∂utx = ∂Ax∂u' * x
    H∂Ax∂utx = solve(LinearProblem(H, ∂Ax∂utx)).u

    # White noise component
    ∂Ax∂ξtx = √(α) * l * pr.L' * H∂Ax∂utx

    # Standard deviation component
    ∂Ax∂σtx = ((u .- pr.μ) / σ)' * ∂Ax∂utx
    ∂Ax∂ωσtx = pr.Δσ * pdf(UNIT_NORM, ω_σ) * ∂Ax∂σtx
    
    # Lengthscale component
    ∂Ax∂ltx = (u - pr.μ)' * (l^-1.0 * pr.M - l * pr.K) * H∂Ax∂utx
    ∂Ax∂ωltx = pr.Δl * pdf(UNIT_NORM, ω_l) * ∂Ax∂ltx

    return vcat(∂Ax∂ξtx, ∂Ax∂ωσtx, ∂Ax∂ωltx)

end

function compute_∂Ax∂θx(
    ∂Ax∂u::AbstractMatrix, 
    θ::AbstractVector, 
    u::AbstractVector, 
    x::AbstractVector,
    pr::MaternField
)::AbstractVector

    ω_σ, ω_l = θ[end-1:end]

    σ = gauss_to_unif(ω_σ, pr.σ_bounds...)
    l = gauss_to_unif(ω_l, pr.l_bounds...)

    α = σ^2 * (4π * gamma(2)) / gamma(1)

    H = pr.M + l^2 * pr.K + l / 1.42 * pr.N 

    # White noise component (TODO: precompute LU factorisation of H?)
    ∂u∂ξx = solve(LinearProblem(H, √(α) * l * pr.L * x[1:end-2])).u
    ∂Ax∂ξx = ∂Ax∂u * ∂u∂ξx

    # Standard deviation component
    ∂σ∂ωσx = pr.Δσ * pdf(UNIT_NORM, ω_σ) * x[end-1]
    ∂Ax∂ωσx = ∂Ax∂u * (u - pr.μ) / σ * ∂σ∂ωσx

    # Lengthscale component
    ∂l∂ωlx = pr.Δl * pdf(UNIT_NORM, ω_l) * x[end]
    ∂u∂ωlx = solve(LinearProblem(H, (l^-1.0 * pr.M - l * pr.K) * (u .- pr.μ) * ∂l∂ωlx)).u
    ∂Ax∂ωlx = ∂Ax∂u * ∂u∂ωlx

    return ∂Ax∂ξx + ∂Ax∂ωσx + ∂Ax∂ωlx

end

function compute_∇Lθ(
    ∂Ap∂u::AbstractMatrix,
    ∂Aμ∂u::AbstractMatrix,
    θ::AbstractVector,
    u::AbstractVector,
    λ::AbstractVector,
    pr::MaternField
)::AbstractVector

    return θ + 
        compute_∂Ax∂θtx(∂Ap∂u, θ, u, λ, pr) + 
        compute_∂Ax∂θtx(∂Aμ∂u, θ, u, λ, pr)

end

function solve_forward_inc(
    Au_r::AbstractMatrix,
    b::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::AbstractVector

    b = reshape(b, m.np_r, g.nt)
    p = zeros(m.np_r, g.nt)

    p[:, 1] = solve(LinearProblem(Au_r, b[:, 1])).u

    for t ∈ 2:g.nt
        bt = b[:, t] + m.ϕ * m.c * p[:, t-1]
        p[:, t] = solve(LinearProblem(Au_r, bt)).u
    end

    return vec(p)

end

function solve_adjoint_inc(
    Au_r::AbstractMatrix,
    b::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)::AbstractVector
    
    Au_rt = Matrix(Au_r')
    b = reshape(b, m.np_r, g.nt)

    λ = zeros(m.np_r, g.nt)
    λ[:, end] = solve(LinearProblem(Au_rt, b[:, end])).u

    for t ∈ (g.nt-1):-1:1
        bt = b[:, t] + m.c * m.ϕ * λ[:, t+1]
        λ[:, t] = solve(LinearProblem(Au_rt, bt)).u
    end

    return vec(λ)

end

function compute_Hmx(
    x::AbstractVector, 
    θ::AbstractVector,
    u::AbstractVector,
    Au_r::AbstractMatrix, 
    ∂Ap∂u::AbstractMatrix,
    ∂Aμ∂u::AbstractMatrix,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField
)::AbstractVector

    ∂Ap∂θx = compute_∂Ax∂θx(∂Ap∂u, θ, u, x, pr)
    ∂Aμ∂θx = compute_∂Ax∂θx(∂Aμ∂u, θ, u, x, pr)

    p_inc = solve_forward_inc(Au_r, ∂Ap∂θx + ∂Aμ∂θx, g, m)
    b_inc = m.BV_r' * m.C_e_inv * m.BV_r * p_inc
    λ_inc = solve_adjoint_inc(Au_r, b_inc, g, m)

    ∂Ap∂θtλ_inc = compute_∂Ax∂θtx(∂Ap∂u, θ, u, λ_inc, pr)
    ∂Aμ∂θtλ_inc = compute_∂Ax∂θtx(∂Aμ∂u, θ, u, λ_inc, pr)

    return ∂Ap∂θtλ_inc + ∂Aμ∂θtλ_inc

end

function compute_Hx(
    x::AbstractVector, 
    θ::AbstractVector,
    u::AbstractVector,
    Au_r::AbstractMatrix, 
    ∂Ap∂u::AbstractMatrix,
    ∂Aμ∂u::AbstractMatrix,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField
)::AbstractVector

    return compute_Hmx(x, θ, u, Au_r, ∂Ap∂u, ∂Aμ∂u, g, m, pr) + x

end

function solve_cg(
    θ::AbstractVector,
    u::AbstractVector,
    Au_r::AbstractMatrix,
    ∂Ap∂u::AbstractMatrix,
    ∂Aμ∂u::AbstractMatrix,
    ∇Lθ::AbstractVector,
    tol::Real,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField
)::AbstractVector

    println("CG It. | norm(r)")
    δθ = spzeros(pr.Nθ)
    d = -copy(∇Lθ)
    r = -copy(∇Lθ)

    i = 1
    while true

        Hd = compute_Hx(d, θ, u, Au_r, ∂Ap∂u, ∂Aμ∂u, g, m, pr)

        α = (r' * r) / (d' * Hd)
        δθ += α * d

        r_prev = copy(r)
        r = r_prev - α * Hd

        @printf "%6i | %.3e\n" i norm(r)

        if norm(r) < tol
            println("CG converged after $i iterations.")
            return δθ
        elseif i > CG_MAX_ITS
            @warn "CG failed to converge within $CG_MAX_ITS iterations."
            return δθ
        end
        
        β = (r' * r) / (r_prev' * r_prev)
        d = r + β * d
        i += 1

    end

end

function linesearch(
    θ_c::AbstractVector, 
    p_r_c::AbstractVector, 
    ∇Lθ_c::AbstractVector, 
    δθ::AbstractVector,
    y::AbstractVector,
    g::Grid, 
    m::ReducedOrderModel,
    pr::MaternField
)::Tuple{AbstractVector, AbstractVector}

    println("LS It. | J(η, u)")
    J_c = J(θ_c, p_r_c, y, g, m)
    
    α_k = 1.0
    θ_k = nothing
    p_r_k = nothing

    i_ls = 1
    while i_ls < LS_MAX_ITS

        θ_k = θ_c + α_k * δθ
        u_k = transform(pr, θ_k)

        Au = m.c * m.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(u_k)) * g.∇h
        Au_r = m.V_ri' * Au * m.V_ri 

        p_r_k = solve_forward(Au, Au_r, g, m)
        J_k = J(θ_k, p_r_k, y, g, m)

        @printf "%6i | %.3e\n" i_ls J_k 

        if (J_k ≤ J_c + LS_C * α_k * ∇Lθ_c' * δθ)
            println("Linesearch converged after $i_ls iterations.")
            return θ_k, p_r_k
        end

        α_k *= 0.5
        i_ls += 1

    end

    @warn "Linesearch failed to converge within $LS_MAX_ITS iterations."
    return θ_k, p_r_k

end

function compute_map(
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField,
    y::AbstractVector,
    θ0::AbstractVector
)::GNResult

    θ = copy(θ0) 
    p_r = nothing
    norm∇Lθ0 = nothing
    
    i = 1
    while true

        @info "Beginning GN It. $i"
        
        u = transform(pr, θ)

        Au = m.c * m.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(u)) * g.∇h
        Au_r = m.V_ri' * Au * m.V_ri

        if i == 1
            p_r = solve_forward(Au, Au_r, g, m)
        end
        λ = solve_adjoint(p_r, Au_r, y, g, m)

        # TODO: how to speed these up?
        ∂Ap∂u = compute_∂Ap∂u(u, p_r, g, m)
        ∂Aμ∂u = compute_∂Aμ∂u(u, g, m)

        ∇Lθ = compute_∇Lθ(∂Ap∂u, ∂Aμ∂u, θ, u, λ, pr)
        if norm(∇Lθ) < GN_MIN_NORM
            @info "Converged."
            return GNResult(true, θ, u, p_r, Au_r, ∂Ap∂u, ∂Aμ∂u)
        end
        
        if i == 1
            norm∇Lθ0 = norm(∇Lθ)
        end
        tol_cg = 0.1 * min(0.5, √(norm(∇Lθ) / norm∇Lθ0)) * norm(∇Lθ)

        @printf "norm(∇Lθ): %.3e\n" norm(∇Lθ)
        @printf "CG tolerance: %.3e\n" tol_cg

        δθ = solve_cg(θ, u, Au_r, ∂Ap∂u, ∂Aμ∂u, ∇Lθ, tol_cg, g, m, pr)
        θ, p_r = linesearch(θ, p_r, ∇Lθ, δθ, y, g, m, pr)
        
        i += 1
        if i > GN_MAX_ITS
            @warn "Failed to converge within $GN_MAX_ITS iterations."
            return GNResult(false, θ, u, p_r, Au_r, ∂Ap∂u, ∂Aμ∂u)
        end

    end

end

function compute_laplace(
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField,
    y::AbstractVector,
    θ0::AbstractVector;
    n_eigvals::Int=30
)::Tuple{GNResult, AbstractMatrix, AbstractMatrix}

    map = compute_map(g, m, pr, y, θ0)
    !map.converged && @warn "MAP optimisation failed to converge."

    f(x) = compute_Hmx(x, map.θ, map.u, map.Au_r, map.∂Ap∂u, map.∂Aμ∂u, g, m, pr)

    vals, vecs, info = eigsolve(f, pr.Nθ, n_eigvals, :LM, issymmetric=true)
    info.converged != length(vals) && @warn "eigsolve did not converge."

    println(info.numops)
    println(info.numiter)
    println(minimum(vals))
    
    λ_r = vals[vals .> 1e-2]
    V_r = hcat(vecs[vals .> 1e-2]...)

    D_r = diagm(λ_r ./ (λ_r .+ 1.0))
    P_r = diagm(1.0 ./ sqrt.(λ_r .+ 1.0) .- 1.0)

    Γ_post = I - V_r * D_r * V_r'
    L_post = V_r * P_r * V_r' + I

    return map, Γ_post, L_post

end

function compute_Jx(
    x::AbstractVector, 
    θ::AbstractVector,
    u::AbstractVector,
    Au_r::AbstractMatrix, 
    ∂Ap∂u::AbstractMatrix,
    ∂Aμ∂u::AbstractMatrix,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField
)

    ∂Ap∂θx = compute_∂Ax∂θx(∂Ap∂u, θ, u, x, pr)
    ∂Aμ∂θx = compute_∂Ax∂θx(∂Aμ∂u, θ, u, x, pr)

    p_inc = solve_forward_inc(Au_r, ∂Ap∂θx + ∂Aμ∂θx, g, m)
    Jx = -m.L_e * m.BV_r * p_inc

    return Jx

end

function compute_JX(
    X::AbstractMatrix, 
    θ::AbstractVector,
    u::AbstractVector,
    Au_r::AbstractMatrix, 
    ∂Ap∂u::AbstractMatrix,
    ∂Aμ∂u::AbstractMatrix,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField
)

    function solve_forward_lu(
        Au_r::LU,
        b::AbstractVector,
        g::Grid,
        m::ReducedOrderModel
    )

        b = reshape(b, m.np_r, g.nt)
        p = zeros(m.np_r, g.nt)

        p[:, 1] = Au_r \ b[:, 1]

        for t ∈ 2:g.nt
            bt = b[:, t] + m.ϕ * m.c * p[:, t-1]
            p[:, t] = Au_r \ bt
        end

        return vec(p)

    end

    JX = zeros(size(m.B, 1), size(X, 2))
    Au_r_fac = lu(Au_r)

    for (i, x) ∈ enumerate(eachcol(X))

        ∂Ap∂θx = compute_∂Ax∂θx(∂Ap∂u, θ, u, x, pr)
        ∂Aμ∂θx = compute_∂Ax∂θx(∂Aμ∂u, θ, u, x, pr)

        p_inc = solve_forward_lu(Au_r_fac, ∂Ap∂θx + ∂Aμ∂θx, g, m)
        JX[:, i] = -m.L_e * m.BV_r * p_inc
    
    end

    return JX

end

function compute_Jtx(
    x::AbstractVector,
    θ::AbstractVector,
    u::AbstractVector,
    Au_r::AbstractMatrix, 
    ∂Ap∂u::AbstractMatrix,
    ∂Aμ∂u::AbstractMatrix,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField
)

    λ = solve_adjoint_inc(Au_r, m.BV_r' * m.L_e' * x, g, m)

    ∂Ap∂θtx = compute_∂Ax∂θtx(∂Ap∂u, θ, u, λ, pr)
    ∂Aμ∂θtx = compute_∂Ax∂θtx(∂Aμ∂u, θ, u, λ, pr)

    Jtx = -∂Ap∂θtx - ∂Aμ∂θtx
    return Jtx

end

G(p, m, y) = m.L_e * (m.B * p + m.μ_e - y)

"""Least-squares RTO functional."""
function J_rto(
    θ::AbstractVector, 
    p_r::AbstractVector,
    y::AbstractVector,
    g::Grid,
    m::ReducedOrderModel,
    Ψ::AbstractMatrix,
    Λ::AbstractMatrix,
    Φ::AbstractMatrix,
    η::AbstractVector
)::Real

    p = get_full_state(p_r, g, m)

    res = (Λ^2 + I)^-0.5 * (Λ * Ψ' * G(p, m, y) + Φ' * (θ - η))
    return 0.5 * sqnorm(res)

end

function compute_∇Lθ_rto(
    ∂Ap∂u::AbstractMatrix,
    ∂Aμ∂u::AbstractMatrix,
    θ::AbstractVector,
    u::AbstractVector,
    p_r::AbstractVector,
    λ::AbstractVector,
    y::AbstractVector,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField,
    Ψ::AbstractMatrix, 
    Λ::AbstractMatrix, 
    Φ::AbstractMatrix,
    η::AbstractVector
)

    p = get_full_state(p_r, g, m)

    Gputλ = compute_∂Ax∂θtx(∂Ap∂u, θ, u, λ, pr) + 
            compute_∂Ax∂θtx(∂Aμ∂u, θ, u, λ, pr)

    ∇Lθ = Φ * (Λ^2 + I)^-1 * (Φ' * θ + (Λ * Ψ' * G(p, m, y) - Φ' * η)) + Gputλ
    return ∇Lθ

end

function compute_Hx_rto(
    x::AbstractVector, 
    θ::AbstractVector,
    u::AbstractVector,
    Au_r::AbstractMatrix, 
    ∂Ap∂u::AbstractMatrix,
    ∂Aμ∂u::AbstractMatrix,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField,
    Λ::AbstractMatrix,
    Φ::AbstractMatrix
)::AbstractVector

    Hx = compute_Hmx(x, θ, u, Au_r, ∂Ap∂u, ∂Aμ∂u, g, m, pr)
    Hx += Φ * (Λ^2 + I)^-1 * Φ' * x
    return Hx

end

function solve_cg_rto(
    θ::AbstractVector,
    u::AbstractVector,
    Au_r::AbstractMatrix,
    ∂Ap∂u::AbstractMatrix,
    ∂Aμ∂u::AbstractMatrix,
    ∇Lθ::AbstractVector,
    tol::Real,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField,
    Λ::AbstractMatrix,
    Φ::AbstractMatrix;
    verbose::Bool=false
)::AbstractVector

    verbose && println("CG It. | norm(r)")
    δθ = spzeros(pr.Nθ)
    d = -copy(∇Lθ)
    r = -copy(∇Lθ)

    i = 1
    while true

        Hd = compute_Hx_rto(d, θ, u, Au_r, ∂Ap∂u, ∂Aμ∂u, g, m, pr, Λ, Φ)

        α = (r' * r) / (d' * Hd)
        δθ += α * d

        r_prev = copy(r)
        r = r_prev - α * Hd

        verbose && @printf "%6i | %.3e\n" i norm(r)

        if norm(r) < tol
            verbose && println("CG converged after $i iterations.")
            return δθ
        elseif i > CG_MAX_ITS
            verbose && @warn "CG failed to converge within $CG_MAX_ITS iterations."
            return δθ
        end
        
        β = (r' * r) / (r_prev' * r_prev)
        d = r + β * d
        i += 1

    end

end

function linesearch_rto(
    θ_c::AbstractVector, 
    p_r_c::AbstractVector, 
    ∇Lθ_c::AbstractVector, 
    δθ::AbstractVector,
    y::AbstractVector,
    g::Grid, 
    m::ReducedOrderModel,
    pr::MaternField,
    Ψ::AbstractMatrix, 
    Λ::AbstractMatrix, 
    Φ::AbstractMatrix, 
    η::AbstractVector;
    verbose::Bool=false
)

    verbose && println("LS It. | J(η, u)")
    J_c = J(θ_c, p_r_c, y, g, m)
    
    α_k = 1.0
    θ_k = nothing
    p_r_k = nothing
    J_k = nothing

    i_ls = 1
    while i_ls < LS_MAX_ITS

        θ_k = θ_c + α_k * δθ
        u_k = transform(pr, θ_k)

        Au = m.c * m.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(u_k)) * g.∇h
        Au_r = m.V_ri' * Au * m.V_ri 

        p_r_k = solve_forward(Au, Au_r, g, m)
        J_k = J_rto(θ_k, p_r_k, y, g, m, Ψ, Λ, Φ, η)

        verbose && @printf "%6i | %.3e\n" i_ls J_k 

        if (J_k ≤ J_c + LS_C * α_k * ∇Lθ_c' * δθ)
            verbose && println("Linesearch converged after $i_ls iterations.")
            return θ_k, p_r_k, J_k
        end

        α_k *= 0.5
        i_ls += 1

    end

    @warn "Linesearch failed to converge within $LS_MAX_ITS iterations."
    return θ_k, p_r_k, J_k

end

function optimise_rto(
    g::Grid,
    m::ReducedOrderModel, 
    pr::MaternField,
    y::AbstractVector,
    θ0::AbstractVector,
    Ψ::AbstractMatrix,
    Λ::AbstractMatrix,
    Φ::AbstractMatrix,
    η::AbstractVector;
    verbose::Bool=false
)

    θ = copy(θ0) 
    p_r = nothing
    norm∇Lθ0 = nothing

    i = 1
    while true

        verbose && @info "Beginning GN It. $i"
        
        u = transform(pr, θ)

        Au = m.c * m.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(u)) * g.∇h
        Au_r = m.V_ri' * Au * m.V_ri

        if i == 1
            p_r = solve_forward(Au, Au_r, g, m)
        end
        p = get_full_state(p_r, g, m)
        b_inc = m.BV_r' * m.L_e' * Ψ * Λ' * (Λ^2 + I)^-1 * (Λ * Ψ' * G(p, m, y) + Φ' * (θ - η))
        λ = solve_adjoint_inc(Au_r, -b_inc, g, m)

        ∂Ap∂u = compute_∂Ap∂u(u, p_r, g, m)
        ∂Aμ∂u = compute_∂Aμ∂u(u, g, m)

        ∇Lθ = compute_∇Lθ_rto(∂Ap∂u, ∂Aμ∂u, θ, u, p_r, λ, y, g, m, pr, Ψ, Λ, Φ, η)
        if i == 1
            norm∇Lθ0 = norm(∇Lθ)
        end
        tol_cg = min(0.5, √(norm(∇Lθ) / norm∇Lθ0)) * norm(∇Lθ)

        verbose && @printf "norm(∇Lθ): %.3e\n" norm(∇Lθ)
        verbose && @printf "CG tolerance: %.3e\n" tol_cg

        δθ = solve_cg_rto(θ, u, Au_r, ∂Ap∂u, ∂Aμ∂u, ∇Lθ, tol_cg, g, m, pr, Λ, Φ; verbose=verbose)
        θ, p_r, Jθ = linesearch_rto(θ, p_r, ∇Lθ, δθ, y, g, m, pr, Ψ, Λ, Φ, η; verbose=verbose)
        
        if Jθ < 1e-5 # TODO: clean up..
            return GNResult(true, θ, u, p_r, Au_r, ∂Ap∂u, ∂Aμ∂u)
        end

        i += 1
        if i > GN_MAX_ITS
            @warn "Failed to converge within $GN_MAX_ITS iterations."
            return GNResult(false, θ, u, p_r, Au_r, ∂Ap∂u, ∂Aμ∂u)
        end

    end

end

function compute_weight_rto(
    sol::GNResult,
    y::AbstractVector,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField,
    Ψ::AbstractMatrix,
    Λ::AbstractMatrix,
    Φ::AbstractMatrix
)

    p = get_full_state(sol.p_r, g, m)

    JX = compute_JX(Φ, sol.θ, sol.u, sol.Au_r, sol.∂Ap∂u, sol.∂Aμ∂u, g, m, pr)
    ldetQH = logabsdet((Λ^2 + I)^-0.5)[1] + logabsdet(I + Λ * Ψ' * JX)[1]
    Gθ = m.L_e * (m.B * p + m.μ_e - y)
    
    logw = - ldetQH
           - 0.5sqnorm(Gθ) 
           - 0.5sqnorm(Φ' * sol.θ) 
           + 0.5sqnorm((Λ^2 + I)^-0.5 * (Φ' * sol.θ + Λ * Ψ' * Gθ))

    return logw

end

function run_rto(
    g::Grid, 
    m::ReducedOrderModel, 
    pr::MaternField,
    y::AbstractVector,
    n_samples::Int;
    n_svd::Int=50,
    verbose::Bool=false
)

    NG = length(y)

    function jac_func(x, flag)
        if flag === Val(true)
            return compute_Jtx(
                x, map.θ, map.u, map.Au_r, 
                map.∂Ap∂u, map.∂Aμ∂u, g, m, pr
            )
        else
            return compute_Jx(
                x, map.θ, map.u, map.Au_r, 
                map.∂Ap∂u, map.∂Aμ∂u, g, m, pr
            )
        end
    end

    θ0 = vec(rand(pr))
    map = compute_map(g, m, pr, y, θ0)

    λs, Ψ, Φ, info = svdsolve(jac_func, NG, n_svd, :LR, krylovdim=100)
    info.converged != length(λs) && @warn "svdsolve did not converge."

    Ψ = hcat(Ψ...)
    Λ = Diagonal(λs)
    Φ = hcat(Φ...)

    println("Min λ: $(minimum(λs))")
    
    samples = [map]
    lnws = [compute_weight_rto(map, y, g, m, pr, Ψ, Λ, Φ)]

    for i ∈ 1:n_samples

        η_i = rand(UNIT_NORM, pr.Nθ)
        θ0_i = vec(rand(pr))

        sample = optimise_rto(g, m, pr, y, θ0_i, Ψ, Λ, Φ, η_i; verbose=verbose)
        lnw = compute_weight_rto(sample, y, g, m, pr, Ψ, Λ, Φ)

        push!(samples, sample)
        push!(lnws, lnw)

        @info "Sample $i computed."

    end

    return samples, lnws

end

function run_rto_mcmc(
    samples::AbstractVector,
    lnws::AbstractVector
)

    n = length(lnws)
    chain = ones(Int, n_samples)
    acc = 0

    for i ∈ 2:n 
        if log(rand()) ≤ (lnws[i] - lnws[chain[i-1]])
            chain[i] = i
            acc += 1
        else
            chain[i] = chain[i-1]
        end
    end

    @info "Acceptance rate: $(acc/n)"

    return samples[chain]

end