using LinearAlgebra
using SparseArrays

"""Builds gradient operator."""
function build_∇h(nx::Real, Δx::Real)::SparseMatrixCSC

    # Inner points (TODO: figure out why centred differences don't seem to work very well)
    ∇hi_i = repeat(2:(nx-1), inner=2)
    ∇hi_j = vcat([[i-1, i+1] for i ∈ 2:(nx-1)]...)
    ∇hi_v = repeat([-1/2, 1/2], outer=(nx-2))

    # Neumann boundaries (TODO: check these)
    push!(∇hi_i, 1, 1, 1, nx, nx, nx)
    push!(∇hi_j, 1, 2, 3, nx-2, nx-1, nx)
    push!(∇hi_v, -3/2, 2, -1/2, 1/2, -2, 3/2)

    ∇hi = sparse(∇hi_i, ∇hi_j, ∇hi_v, nx, nx) / Δx
    Ii = sparse(I, nx, nx)

    ∇h = [kron(Ii, ∇hi); kron(∇hi, Ii)]
    return ∇h

end

"""Builds operator that duplicates permeabilities."""
function build_A(nx::Real)::SparseMatrixCSC

    return [sparse(I, nx^2, nx^2); sparse(I, nx^2, nx^2)]

end

struct Grid 

    xs::AbstractVector
    ts::AbstractVector

    cxs::AbstractVector 
    cys::AbstractVector
    
    Δx::Real 
    Δt::Real

    nx::Real 
    nt::Real 

    ∇h::SparseMatrixCSC
    A::SparseMatrixCSC

    ϕ::Real 
    μ::Real
    c::Real
    u0::Real

    function Grid(
        xmax::Real,
        tmax::Real,
        Δx::Real,
        Δt::Real,
        ϕ::Real=1.0,
        μ::Real=1.0,
        c::Real=1.0,
        u0::Real=1.0
    )

        nx = Int(round(xmax / Δx))+1
        nt = Int(round(tmax / Δt))+1

        xs = LinRange(0, xmax, nx)
        ts = LinRange(0, tmax, nt)

        cxs = repeat(xs, outer=nx)
        cys = repeat(xs, inner=nx)

        ∇h = build_∇h(nx, Δx)
        A = build_A(nx)

        return new(xs, ts, cxs, cys, Δx, Δt, nx, nt, ∇h, A, ϕ, μ, c, u0)

    end

end