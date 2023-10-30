using LinearAlgebra
using LinearSolve
using HDF5
using SparseArrays

function SciMLBase.solve(
    g::Grid, 
    m::Model,
    θ::AbstractVector
)::AbstractVector

    u = zeros(g.nx^2, g.nt)

    Aθ = m.ϕ * m.c * sparse(I, g.nx^2, g.nx^2) + 
        (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h
    
    b = g.Δt * m.Q[:, 1] .+ m.ϕ * m.c * m.u0
    u[:, 1] = solve(LinearProblem(Aθ, b))

    for t ∈ 2:g.nt
        b = g.Δt * m.Q[:, t] + m.ϕ * m.c * u[:, t-1]
        u[:, t] = solve(LinearProblem(Aθ, b))
    end

    return vec(u)

end

function SciMLBase.solve(
    g::Grid, 
    m::AbstractModel,
    θ::AbstractVector,
    nu_r::Int,
    μ_ui::AbstractVector,
    V_ri::AbstractMatrix
)::AbstractVector

    u_r = zeros(nu_r, g.nt)

    Aθ = m.ϕ * m.c * sparse(I, g.nx^2, g.nx^2) + 
        (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h
    Aθ_r = V_ri' * Aθ * V_ri

    b = V_ri' * (g.Δt * m.Q[:, 1] .+ (m.ϕ * m.c * m.u0) .- Aθ * μ_ui)
    u_r[:, 1] = solve(LinearProblem(Aθ_r, b))

    for t ∈ 2:g.nt
        bt = V_ri' * (g.Δt * m.Q[:, t] - Aθ * μ_ui) + m.ϕ * m.c * (u_r[:, t-1] + V_ri' * μ_ui)
        u_r[:, t] = solve(LinearProblem(Aθ_r, bt))
    end

    return vec(V_ri * u_r .+ μ_ui)

end

SciMLBase.solve(g::Grid, m::ReducedOrderModel, θ::AbstractVector) = 
    SciMLBase.solve(g, m, θ, m.nu_r, m.μ_ui, m.V_ri)

function generate_pod_samples(
    g::Grid,
    m::Model,
    pr::MaternField,
    N::Int
)::AbstractMatrix

    ηs = rand(pr, N)
    θs = [transform(pr, η) for η ∈ eachcol(ηs)]
    us = hcat([solve(g, m, θ) for θ ∈ θs]...)
    return us

end

function compute_pod_basis(
    g::Grid,
    us::AbstractMatrix,
    var_to_retain::Real
)::Tuple{Int, AbstractVector, AbstractMatrix}

    us_reshaped = reshape(us, g.nx^2, :)'

    μ = vec(mean(us_reshaped, dims=1))
    Γ = cov(us_reshaped)

    eigendecomp = eigen(Γ, sortby=(λ -> -λ))
    Λ, V = eigendecomp.values, eigendecomp.vectors

    N_r = findfirst(cumsum(Λ)/sum(Λ) .> var_to_retain)
    V_r = V[:, 1:N_r]
    @info "Reduced basis computed (dimension: $N_r)."

    return N_r, μ, V_r

end

function compute_error_statistics(
    g::Grid,
    m::Model,
    pr::MaternField,
    nu_r::Int,
    μ_ui::AbstractVector,
    V_ri::AbstractMatrix,
    N::Int
)::Tuple{AbstractVector, AbstractMatrix}

    ηs = rand(pr, N)
    θs = [transform(pr, η) for η ∈ eachcol(ηs)]

    us = [@time solve(g, m, θ) for θ ∈ θs]
    us_r = [@time solve(g, m, θ, nu_r, μ_ui, V_ri) for θ ∈ θs]

    ys = hcat([m.B * u for u ∈ us]...)
    ys_r = hcat([m.B * u for u ∈ us_r]...)

    μ_ε = vec(mean(ys - ys_r, dims=2))
    Γ_ε = cov(ys' - ys_r')

    return μ_ε, Γ_ε

end

function generate_pod_data(
    g::Grid,
    m::Model,
    pr::MaternField,
    N::Int, 
    var_to_retain::Real,
    fname::AbstractString
)

    us_samp = generate_pod_samples(g, m, pr, N)
    nu_r, μ_ui, V_ri = compute_pod_basis(g, us_samp, var_to_retain)
    μ_ε, Γ_ε = compute_error_statistics(g, m, pr, nu_r, μ_ui, V_ri, N)

    h5write("data/$fname.h5", "μ_ui", μ_ui)
    h5write("data/$fname.h5", "V_ri", V_ri)
    h5write("data/$fname.h5", "μ_ε", μ_ε)
    h5write("data/$fname.h5", "Γ_ε", Γ_ε)

    return μ_ui, V_ri, μ_ε, Γ_ε
    
end

function read_pod_data(
    fname::AbstractString
)

    f = h5open("data/$fname.h5")

    μ_ui = f["μ_ui"][:]
    V_ri = f["V_ri"][:, :]
    μ_ε = f["μ_ε"][:]
    Γ_ε = f["Γ_ε"][:, :]

    return μ_ui, V_ri, μ_ε, Γ_ε

end