using LinearAlgebra
using LinearSolve
using SparseArrays

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