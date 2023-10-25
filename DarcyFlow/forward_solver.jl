using LinearAlgebra
using LinearSolve
using HDF5
using SparseArrays

"""Well that has been mollified using a bump function of radius r."""
struct Well

    cx::Real
    cy::Real
    r::Real

    rates::Tuple
    Z::Real
    
    function Well(
        g::Grid, 
        cx::Real, 
        cy::Real, 
        r::Real,
        rates::Tuple
    )::Well

        """Calculates the value to scale the bump function by, such that 
        the values of the function on the model grid sum to 1."""
        function normalising_constant(
            g::Grid, 
            cx::Real, 
            cy::Real, 
            r::Real
        )::Real
        
            Z = 0.0
            for (x, y) ∈ zip(g.cxs, g.cys)
                if (r_sq = (x-cx)^2 + (y-cy)^2) < r^2
                    Z += exp(-1/(r^2-r_sq))
                end
            end
        
            return Z
        
        end

        Z = normalising_constant(g, cx, cy, r)

        return new(cx, cy, r, rates, Z)
    
    end

end

"""Builds the forcing term at each time index."""
function build_Q(
    g::Grid,
    wells::AbstractVector{Well},
    well_change_times::AbstractVector 
)::SparseMatrixCSC

    Q_i = Int[]
    Q_j = Int[]
    Q_v = Float64[]

    time_inds = [findlast(well_change_times .<= t + 1e-8) for t ∈ g.ts]

    for (i, (x, y)) ∈ enumerate(zip(g.cxs, g.cys))
        for w ∈ wells 
            if (dist_sq = (x-w.cx)^2 + (y-w.cy)^2) < w.r^2
                for (j, q) ∈ enumerate(w.rates[time_inds])
                    push!(Q_i, i)
                    push!(Q_j, j)
                    push!(Q_v, q * exp(-1/(q^2-dist_sq)) / w.Z)
                end
            end
        end
    end

    Q = sparse(Q_i, Q_j, Q_v, g.nx^2, g.nt+1)
    return Q

end

"""Solves the full model."""
function SciMLBase.solve(
    g::Grid, 
    θ::AbstractVector, 
    Q::AbstractMatrix
)::AbstractVector

    u = zeros(g.nx^2, g.nt)

    A = (1.0 / g.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h 
    B = g.ϕ * g.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * A 
    
    prob = LinearProblem(B, g.Δt*Q[:, 1] .+ g.ϕ*g.c*g.u0)
    u[:, 1] = solve(prob)

    for t ∈ 2:g.nt
        prob = LinearProblem(B, g.Δt*Q[:, t] + g.ϕ*g.c*u[:, t-1])
        u[:, t] = solve(prob)
    end

    return vec(u)

end

"""Solves the reduced-order model."""
function SciMLBase.solve(
    g::Grid, 
    θ::AbstractVector, 
    Q::AbstractMatrix,
    μ_u::AbstractVector,
    V_r::AbstractMatrix
)::AbstractVector

    nu_r = size(V_r, 2)
    u = zeros(nu_r, g.nt)

    Aθ = (1.0 / g.μ) * g.∇h' * spdiagm(g.A * exp.(θ)) * g.∇h
    Bθ = g.ϕ * g.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * Aθ
    B̃θ = V_r' * Bθ * V_r

    b = V_r' * (g.Δt * Q[:, 1] .+ (g.ϕ * g.c * u0) .- Bθ * μ_u)
    u[:, 1] = solve(LinearProblem(B̃θ, b))

    for t ∈ 2:g.nt
        b = V_r' * (g.Δt * Q[:, t] + g.ϕ * g.c * (V_r * u[:, t-1] + μ_u) - Bθ * μ_u)
        u[:, t] = solve(LinearProblem(B̃θ, b))
    end

    return vec(V_r * u .+ μ_u)

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