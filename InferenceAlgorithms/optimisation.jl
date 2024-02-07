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

    n_solves::Int

end

sqnorm(x) = sum(x.^2)

function get_full_state(
    p_r::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)
    
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
)

    p = get_full_state(p_r, g, m)
    res = m.B * p + m.μ_e - y
    return 0.5 * res' * m.C_e_inv * res + 0.5 * sum(θ.^2)

end

function compute_∂Ap∂u(
    u::AbstractVector, 
    p_r::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)

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
) 

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
)

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
)

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
)

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
)

    ω_σ, ω_l = θ[end-1:end]

    σ = gauss_to_unif(ω_σ, pr.σ_bounds...)
    l = gauss_to_unif(ω_l, pr.l_bounds...)

    α = σ^2 * (4π * gamma(2)) / gamma(1)

    H = pr.M + l^2 * pr.K + l / 1.42 * pr.N 

    # White noise component
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
)

    return θ + 
        compute_∂Ax∂θtx(∂Ap∂u, θ, u, λ, pr) + 
        compute_∂Ax∂θtx(∂Aμ∂u, θ, u, λ, pr)

end

function solve_forward_inc(
    Au_r::AbstractMatrix,
    b::AbstractVector,
    g::Grid,
    m::ReducedOrderModel
)

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
)
    
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
)

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
)

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
)

    println("CG It. | norm(r)")
    δθ = spzeros(pr.Nθ)
    d = -copy(∇Lθ)
    r = -copy(∇Lθ)

    i = 0
    while true

        i += 1

        Hd = compute_Hx(d, θ, u, Au_r, ∂Ap∂u, ∂Aμ∂u, g, m, pr)

        α = (r' * r) / (d' * Hd)
        δθ += α * d

        r_prev = copy(r)
        r = r_prev - α * Hd

        @printf "%6i | %.3e\n" i norm(r)

        if norm(r) < tol
            println("CG converged after $i iterations.")
            return δθ, i
        elseif i > CG_MAX_ITS
            @warn "CG failed to converge within $CG_MAX_ITS iterations."
            return δθ, i
        end
        
        β = (r' * r) / (r_prev' * r_prev)
        d = r + β * d

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
)

    println("LS It. | J(η, u)")
    J_c = J(θ_c, p_r_c, y, g, m)
    
    α_k = 1.0
    θ_k = nothing
    p_r_k = nothing

    i = 0
    while i < LS_MAX_ITS

        i += 1

        θ_k = θ_c + α_k * δθ
        u_k = transform(pr, θ_k)

        Au = m.c * m.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(u_k)) * g.∇h
        Au_r = m.V_ri' * Au * m.V_ri 

        p_r_k = solve_forward(Au, Au_r, g, m)
        J_k = J(θ_k, p_r_k, y, g, m)

        @printf "%6i | %.3e\n" i J_k 

        if (J_k ≤ J_c + LS_C * α_k * ∇Lθ_c' * δθ)
            println("Linesearch converged after $i iterations.")
            return θ_k, p_r_k, i
        end

        α_k *= 0.5

    end

    @warn "Linesearch failed to converge within $LS_MAX_ITS iterations."
    return θ_k, p_r_k, i

end

function compute_map(
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField,
    y::AbstractVector,
    θ0::AbstractVector
)

    θ = copy(θ0) 
    p_r = nothing
    norm∇Lθ0 = nothing
    
    i = 0
    n_solves = 0
    while true

        i += 1

        @info "Beginning GN It. $i"
        
        u = transform(pr, θ)

        Au = m.c * m.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(u)) * g.∇h
        Au_r = m.V_ri' * Au * m.V_ri

        if i == 1
            p_r = solve_forward(Au, Au_r, g, m)
            n_solves += 1
        end
        λ = solve_adjoint(p_r, Au_r, y, g, m)
        n_solves += 1

        ∂Ap∂u = compute_∂Ap∂u(u, p_r, g, m)
        ∂Aμ∂u = compute_∂Aμ∂u(u, g, m)

        ∇Lθ = compute_∇Lθ(∂Ap∂u, ∂Aμ∂u, θ, u, λ, pr)
        if i == 1
            norm∇Lθ0 = norm(∇Lθ)
        end
        tol_cg = 0.1 * min(0.5, √(norm(∇Lθ) / norm∇Lθ0)) * norm(∇Lθ)

        if norm(∇Lθ) < 1e-5 * norm∇Lθ0
            @info "Converged."
            return GNResult(true, θ, u, p_r, Au_r, ∂Ap∂u, ∂Aμ∂u, n_solves)
        end

        @printf "norm(∇Lθ): %.3e\n" norm(∇Lθ)
        @printf "CG tolerance: %.3e\n" tol_cg

        δθ, n_it_cg = solve_cg(θ, u, Au_r, ∂Ap∂u, ∂Aμ∂u, ∇Lθ, tol_cg, g, m, pr)
        θ, p_r, n_it_ls = linesearch(θ, p_r, ∇Lθ, δθ, y, g, m, pr)
        
        n_solves += (2n_it_cg + n_it_ls)

        if i > GN_MAX_ITS
            @warn "Failed to converge within $GN_MAX_ITS iterations."
            return GNResult(false, θ, u, p_r, Au_r, ∂Ap∂u, ∂Aμ∂u, n_solves)
        end

    end

end

function compute_laplace(
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField,
    y::AbstractVector,
    θ0::AbstractVector;
    n_eigvals::Int=28
)

    map = compute_map(g, m, pr, y, θ0)
    !map.converged && @warn "MAP optimisation failed to converge."

    f(x) = compute_Hmx(x, map.θ, map.u, map.Au_r, map.∂Ap∂u, map.∂Aμ∂u, g, m, pr)

    vals, vecs, info = eigsolve(f, pr.Nθ, n_eigvals, :LM, issymmetric=true)
    info.converged != length(vals) && @warn "eigsolve did not converge."
    
    λ_r = vals[vals .> 1e-2]
    V_r = hcat(vecs[vals .> 1e-2]...)

    D_r = diagm(λ_r ./ (λ_r .+ 1.0))
    P_r = diagm(1.0 ./ sqrt.(λ_r .+ 1.0) .- 1.0)

    Γ_post = I - V_r * D_r * V_r'
    L_post = V_r * P_r * V_r' + I

    n_solves = map.n_solves + 2 * info.numops

    @info "Minimum eigenvalue: $(minimum(vals))"
    @info "Total solves: $(n_solves)"
    
    return map, Γ_post, L_post, n_solves

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
    θ_r::AbstractVector, 
    p_r::AbstractVector,
    y::AbstractVector,
    g::Grid,
    m::ReducedOrderModel,
    Ψ::AbstractMatrix,
    Λ::AbstractMatrix,
    Φ::AbstractMatrix,
    η::AbstractVector
)

    p = get_full_state(p_r, g, m)

    res = (Λ^2 + I)^-0.5 * (θ_r + Λ * Ψ' * G(p, m, y) - Φ' * η)
    return 0.5 * sqnorm(res)

end

function compute_∇Lθ_r(
    ∂Ap∂u::AbstractMatrix,
    ∂Aμ∂u::AbstractMatrix,
    θ_r::AbstractVector,
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

    Gputλ = Φ' * compute_∂Ax∂θtx(∂Ap∂u, θ, u, λ, pr) + 
            Φ' * compute_∂Ax∂θtx(∂Aμ∂u, θ, u, λ, pr)

    ∇Lθ_r = (Λ^2 + I)^-1 * (θ_r + Λ * Ψ' * G(p, m, y) - Φ' * η) + Gputλ
    return ∇Lθ_r

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
    Ψ::AbstractMatrix,
    Λ::AbstractMatrix,
    Φ::AbstractMatrix
)

    ∂Ap∂θx = compute_∂Ax∂θx(∂Ap∂u, θ, u, Φ * x, pr)
    ∂Aμ∂θx = compute_∂Ax∂θx(∂Aμ∂u, θ, u, Φ * x, pr)

    p_inc = solve_forward_inc(Au_r, ∂Ap∂θx + ∂Aμ∂θx, g, m)
    b_inc = m.BV_r' * m.L_e' * Ψ * Λ * (Λ+I)^-1 * (Λ * Ψ' * L_e * m.BV_r * p_inc - x)
    λ_inc = solve_adjoint_inc(Au_r, b_inc, g, m)

    ∂Ap∂θtλ_inc = Φ' * compute_∂Ax∂θtx(∂Ap∂u, θ, u, λ_inc, pr)
    ∂Aμ∂θtλ_inc = Φ' * compute_∂Ax∂θtx(∂Aμ∂u, θ, u, λ_inc, pr)

    Hx = -(Λ+I)^-1 * Λ * Ψ' * L_e * m.BV_r * p_inc + 
        (Λ+I)^-1 * x + 
        (∂Ap∂θtλ_inc + ∂Aμ∂θtλ_inc)

    return Hx

end

function solve_cg_rto(
    θ::AbstractVector,
    u::AbstractVector,
    Au_r::AbstractMatrix,
    ∂Ap∂u::AbstractMatrix,
    ∂Aμ∂u::AbstractMatrix,
    ∇Lθ_r::AbstractVector,
    tol::Real,
    g::Grid,
    m::ReducedOrderModel,
    pr::MaternField,
    Ψ::AbstractMatrix,
    Λ::AbstractMatrix,
    Φ::AbstractMatrix;
    verbose::Bool=false
)::AbstractVector

    verbose && println("CG It. | norm(r)")
    δθ_r = zeros(size(Φ, 2))
    d = -copy(∇Lθ_r)
    r = -copy(∇Lθ_r)

    i = 1
    while true

        Hd = compute_Hx_rto(d, θ, u, Au_r, ∂Ap∂u, ∂Aμ∂u, g, m, pr, Ψ, Λ, Φ)

        α = (r' * r) / (d' * Hd)
        δθ_r += α * d

        r_prev = copy(r)
        r = r_prev - α * Hd

        verbose && @printf "%6i | %.3e\n" i norm(r)

        if norm(r) < tol
            verbose && println("CG converged after $i iterations.")
            return δθ_r
        elseif i > CG_MAX_ITS
            verbose && @warn "CG failed to converge within $CG_MAX_ITS iterations."
            return δθ_r
        end
        
        β = (r' * r) / (r_prev' * r_prev)
        d = r + β * d
        i += 1

    end

end

function linesearch_rto(
    θ_r_c::AbstractVector, 
    p_r_c::AbstractVector, 
    ∇Lθ_r_c::AbstractVector, 
    δθ_r::AbstractVector,
    y::AbstractVector,
    g::Grid, 
    m::ReducedOrderModel,
    pr::MaternField,
    Ψ::AbstractMatrix, 
    Λ::AbstractMatrix, 
    Φ::AbstractMatrix, 
    θ_c::AbstractVector,
    η::AbstractVector;
    verbose::Bool=false
)

    verbose && println("LS It. | J(η, u)")
    J_c = J(θ_r_c, p_r_c, y, g, m)
    
    α_k = 1.0
    θ_r_k = nothing
    p_r_k = nothing
    J_k = nothing

    i_ls = 1
    while i_ls < LS_MAX_ITS

        θ_r_k = θ_r_c + α_k * δθ_r
        θ_k = Φ * θ_r_k + θ_c
        u_k = transform(pr, θ_k)

        Au = m.c * m.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(u_k)) * g.∇h
        Au_r = m.V_ri' * Au * m.V_ri 

        p_r_k = solve_forward(Au, Au_r, g, m)
        J_k = J_rto(θ_r_k, p_r_k, y, g, m, Ψ, Λ, Φ, η)

        verbose && @printf "%6i | %.3e\n" i_ls J_k 

        if (J_k ≤ J_c + LS_C * α_k * ∇Lθ_r_c' * δθ_r)
            verbose && println("Linesearch converged after $i_ls iterations.")
            return θ_r_k, p_r_k, J_k
        end

        α_k *= 0.5
        i_ls += 1

    end

    @warn "Linesearch failed to converge within $LS_MAX_ITS iterations."
    return θ_r_k, p_r_k, J_k

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

    θ_r = Φ' * θ0
    p_r = nothing
    norm∇Lθ_r0 = nothing
    Jθ = Inf

    θ_c = (I - Φ * Φ') * η

    i = 1
    while true

        verbose && @info "Beginning GN It. $i"
        
        θ = Φ * θ_r + θ_c
        u = transform(pr, θ)

        Au = m.c * m.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(u)) * g.∇h
        Au_r = m.V_ri' * Au * m.V_ri

        if i == 1
            p_r = solve_forward(Au, Au_r, g, m)
        end
        p = get_full_state(p_r, g, m)
        b_inc = m.BV_r' * m.L_e' * Ψ * Λ * (Λ^2 + I)^-1 * (θ_r + Λ * Ψ' * G(p, m, y) - Φ' * η)
        λ = solve_adjoint_inc(Au_r, -b_inc, g, m)

        ∂Ap∂u = compute_∂Ap∂u(u, p_r, g, m)
        ∂Aμ∂u = compute_∂Aμ∂u(u, g, m)

        if Jθ < 1e-5 # TODO: clean up... I don't think I have the right derivatives etc in here.
            return GNResult(true, θ, u, p_r, Au_r, ∂Ap∂u, ∂Aμ∂u, -1) # TODO: add number of solves
        end

        ∇Lθ_r = compute_∇Lθ_r(∂Ap∂u, ∂Aμ∂u, θ_r, θ, u, p_r, λ, y, g, m, pr, Ψ, Λ, Φ, η)
        if i == 1
            norm∇Lθ_r0 = norm(∇Lθ_r)
        end
        tol_cg = 1e-2 * min(0.5, √(norm(∇Lθ_r) / norm∇Lθ_r0)) * norm(∇Lθ_r)

        verbose && @printf "norm(∇Lθ): %.3e\n" norm(∇Lθ_r)
        verbose && @printf "CG tolerance: %.3e\n" tol_cg

        δθ_r = solve_cg_rto(θ, u, Au_r, ∂Ap∂u, ∂Aμ∂u, ∇Lθ_r, tol_cg, g, m, pr, Ψ, Λ, Φ; verbose=verbose)
        θ_r, p_r, Jθ = linesearch_rto(θ_r, p_r, ∇Lθ_r, δθ_r, y, g, m, pr, Ψ, Λ, Φ, θ_c, η; verbose=verbose)

        i += 1
        # if i >  GN_MAX_ITS
        #     @warn "Failed to converge within $GN_MAX_ITS iterations."
        #     return GNResult(false, θ_r, u, p_r, Au_r, ∂Ap∂u, ∂Aμ∂u, -1) # TODO: add number of solves
        # end

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

    JΦ = compute_JX(Φ, sol.θ, sol.u, sol.Au_r, sol.∂Ap∂u, sol.∂Aμ∂u, g, m, pr)
    ldetQH = logabsdet((Λ^2 + I)^-0.5)[1] + logabsdet(I + Λ * Ψ' * JΦ)[1]
    Gθ = G(p, m, y)
    
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

        sample = optimise_rto(g, m, pr, y, map.θ, Ψ, Λ, Φ, η_i; verbose=verbose)
        lnw = compute_weight_rto(sample, y, g, m, pr, Ψ, Λ, Φ)
        
        println(lnw)

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
    chain = ones(Int, length(lnws))
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