using LinearAlgebra
using LinearSolve
using HDF5
using SparseArrays

"""Solves the full model."""
function SciMLBase.solve(
    g::Grid, 
    m::Model,
    θ::AbstractVector
)::AbstractVector

    u = zeros(g.nx^2, g.nt)

    A = (1.0 / m.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h 
    B = m.ϕ * m.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * A 
    
    prob = LinearProblem(B, g.Δt * m.Q[:, 1] .+ m.ϕ * m.c * m.u0)
    u[:, 1] = solve(prob)

    for t ∈ 2:g.nt
        prob = LinearProblem(B, g.Δt * m.Q[:, t] + m.ϕ * m.c * u[:, t-1])
        u[:, t] = solve(prob)
    end

    return vec(u)

end

"""Solves the reduced-order model."""
function SciMLBase.solve(
    g::Grid, 
    m::ReducedOrderModel,
    θ::AbstractVector
)::AbstractVector

    u = zeros(m.nu_r, g.nt)

    Aθ = (1.0 / m.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h
    Bθ = m.ϕ * m.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * Aθ
    B̃θ = m.V_r' * Bθ * m.V_r

    b = m.V_r' * (g.Δt * m.Q[:, 1] .+ (m.ϕ * m.c * m.u0) .- Bθ * m.μ_u)
    u[:, 1] = solve(LinearProblem(B̃θ, b))

    for t ∈ 2:g.nt
        b = m.V_r' * (g.Δt * m.Q[:, t] + m.ϕ * m.c * (m.V_r * u[:, t-1] + m.μ_u) - Bθ * m.μ_u)
        u[:, t] = solve(LinearProblem(B̃θ, b))
    end

    return vec(m.V_r * u .+ m.μ_u)

end

function generate_pod_samples(
    pr::MaternField,
    N::Int
)::AbstractMatrix

    θs = rand(pr, N)
    us = hcat([@time F(θ) for θ ∈ eachcol(θs)]...)

    return us

end

function compute_pod_basis(
    g::Grid,
    us::AbstractMatrix,
    var_to_retain::Real
)::Tuple{AbstractVector, AbstractMatrix}

    us_reshaped = reshape(us, g.nx^2, :)'

    μ = vec(mean(us_reshaped, dims=1))
    Γ = cov(us_reshaped)

    eigendecomp = eigen(Γ, sortby=(λ -> -λ))
    Λ, V = eigendecomp.values, eigendecomp.vectors

    N_r = findfirst(cumsum(Λ)/sum(Λ) .> var_to_retain)
    V_r = V[:, 1:N_r]
    @info "Reduced basis computed (dimension: $N_r)."

    return μ, V_r

end

function compute_error_statistics(
    F::Function,
    G::Function,
    pr::MaternField,
    μ_u::AbstractVector,
    V_r::AbstractMatrix,
    N::Int
)::Tuple{AbstractVector, AbstractMatrix}

    # HACK! TODO: tidy up
    function F_r(η, μ_u, V_r)
        θ = transform(pr, η)
        return solve(grid_c, θ, Q_c, μ_u, V_r)
    end

    ηs = rand(pr, N)
    us = [@time F(η) for η ∈ eachcol(ηs)]
    us_r = [@time F_r(η, μ_u, V_r) for η ∈ eachcol(ηs)]

    ys = hcat([G(u) for u ∈ us]...)
    ys_r = hcat([G(u) for u ∈ us_r]...)

    # Compute statistics of errors
    μ_e = vec(mean(ys - ys_r, dims=2))
    Γ_e = cov(ys' - ys_r')

    return μ_e, Γ_e

end

function generate_pod_data(
    g::Grid,
    F::Function,
    G::Function,
    pr::MaternField,
    N::Int, 
    var_to_retain::Real,
    fname::AbstractString
)

    us_samp = generate_pod_samples(pr, N)
    μ_u, V_r = compute_pod_basis(g, us_samp, var_to_retain)
    μ_ε, Γ_ε = compute_error_statistics(F, G, pr, μ_u, V_r, N)

    h5write("data/$fname.h5", "μ_u", μ_u)
    h5write("data/$fname.h5", "V_r", V_r)
    h5write("data/$fname.h5", "μ_ε", μ_ε)
    h5write("data/$fname.h5", "Γ_ε", Γ_ε)

    return μ_u, V_r, μ_ε, Γ_ε
    
end

function read_pod_data(
    fname::AbstractString
)

    f = h5open("data/$fname.h5")

    μ_u = f["μ_u"][:]
    V_r = f["V_r"][:, :]
    μ_ε = f["μ_ε"][:]
    Γ_ε = f["Γ_ε"][:, :]

    return μ_u, V_r, μ_ε, Γ_ε

end