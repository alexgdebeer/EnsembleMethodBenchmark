using LinearAlgebra
using LinearSolve
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

    A = (1.0 / g.μ) * g.∇h' * spdiagm((g.A * exp.(-θ)) .^ -1) * g.∇h 
    B = g.ϕ * g.c * sparse(I, g.nx^2, g.nx^2) + g.Δt * A 
    
    # Initial solve
    b = g.Δt * Q[:, 1] .+ g.ϕ * g.c * g.u0
    u[:, 1] = solve(LinearProblem(B, b))

    for t ∈ 2:g.nt
        b = g.Δt * Q[:, t] + g.ϕ * g.c * u[:, t-1]
        u[:, t] = solve(LinearProblem(B, b))
    end

    return vec(u)

end

"""Solves the reduced-order model."""
function SciMLBase.solve(
    g::Grid, 
    lnps::AbstractVecOrMat, 
    Q::AbstractMatrix,
    μ::AbstractVector,
    V_r::AbstractMatrix
)::AbstractVector

    us = zeros(g.nx^2, g.nt)
    us[:, 1] .= g.u0

    A = (g.Δt / g.μ) * g.∇h' * spdiagm((g.A * exp.(-vec(lnps))) .^ -1) * g.∇h
    Id = g.ϕ * g.c * sparse(I, g.nx^2, g.nx^2)

    M = Id + A 
    M_r = V_r' * M * V_r

    for t ∈ 1:(g.nt-1)

        b_r = V_r' * (g.Δt * Q[:, t+1] + g.ϕ * g.c * us[:, t] - M * μ)

        us_r = solve(LinearProblem(M_r, b_r))
        us[:, t+1] = μ + V_r * us_r

    end

    return vec(us)

end

function generate_pod_samples(
    p,
    N::Int
)::AbstractMatrix

    θs = rand(p, N)
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