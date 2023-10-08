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

"""Builds the operator that maps from model states to observations."""
function build_B(
    g::Grid,
    x_obs::AbstractVector,
    y_obs::AbstractVector
)::SparseMatrixCSC

    function get_cell_index(g::Grid, xi::Int, yi::Int)
        return xi + g.nx * yi # TODO: check this...
    end
        
    n_obs = length(x_obs)

    is = Int[]
    js = Int[]
    vs = Float64[]

    for (i, (x, y)) ∈ enumerate(zip(x_obs, y_obs))

        ix0 = findfirst(g.xs .> x) - 1
        iy0 = findfirst(g.xs .> y) - 1
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        x0, x1 = g.xs[ix0], g.xs[ix1]
        y0, y1 = g.xs[iy0], g.xs[iy1]

        inds = [(ix0, iy0), (ix0, iy1), (ix1, iy0), (ix1, iy1)]
        cell_inds = [get_cell_index(g, i...) for i ∈ inds]

        Z = (x1-x0) * (y1-y0)

        push!(is, i, i, i, i)
        push!(js, cell_inds...)
        push!(vs,
            (x1-x) * (y1-y) / Z, 
            (x-x0) * (y1-y) / Z, 
            (x1-x) * (y-y0) / Z, 
            (x-x0) * (y-y0) / Z
        )

    end

    return sparse(is, js, vs, n_obs, g.nx^2)

end

"""Solves the full model."""
function SciMLBase.solve(
    g::Grid, 
    logps::AbstractVector, 
    Q::AbstractMatrix
)::AbstractVector

    us = zeros(g.nx^2, g.nt+1)
    us[:, 1] .= g.u0

    A = -g.∇h' * (1/g.μ) * spdiagm(g.A * 10 .^ logps) * g.∇h
    Id = (g.ϕ * g.c / g.Δt) * sparse(I, g.nx^2, g.nx^2)
    M = Id - A

    for t ∈ 1:g.nt
        b = Q[:, t+1] + (g.ϕ * g.c / g.Δt) * us[:, t]
        us[:, t+1] = solve(LinearProblem(M, b))
    end

    return vec(us)

end

"""Solves the reduced-order model."""
function SciMLBase.solve(
    g::Grid, 
    logps::AbstractVector, 
    Q::AbstractMatrix,
    μ::AbstractVector,
    V_r::AbstractMatrix
)::AbstractVector

    us = zeros(g.nx^2, g.nt+1)
    us[:, 1] .= g.u0

    A = -g.∇h' * (1/g.μ) * spdiagm(g.A * 10 .^ logps) * g.∇h
    Id = (g.ϕ * g.c / g.Δt) * sparse(I, g.nx^2, g.nx^2)

    M = Id - A 
    M_r = V_r' * M * V_r

    for t ∈ 1:g.nt

        b_r = V_r' * (Q[:, t+1] + (g.ϕ * g.c / g.Δt) * us[:, t] - M * μ)

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