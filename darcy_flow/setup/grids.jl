using LinearAlgebra
using SparseArrays

function build_∇h(nx::Real, Δx::Real)::SparseMatrixCSC

    # Inner points
    ∇hi_i = repeat(2:nx, inner=2)
    ∇hi_j = vcat([[i-1, i] for i ∈ 2:nx]...)
    ∇hi_v = repeat([-1, 1], outer=(nx-1))

    # Neumann boundaries (TODO: check)
    push!(∇hi_i, 1, 1, nx+1, nx+1)
    push!(∇hi_j, 1, 2, nx-1, nx)
    push!(∇hi_v, -1, 1, -1, 1)

    ∇hi = sparse(∇hi_i, ∇hi_j, ∇hi_v, nx+1, nx) / Δx
    Ii = sparse(I, nx, nx)

    ∇h = [kron(Ii, ∇hi); kron(∇hi, Ii)]
    return ∇h

end

function build_A(nx::Real)::SparseMatrixCSC

    Ai_i = repeat(2:nx, inner=2)
    Ai_j = vcat([[i-1, i] for i ∈ 2:nx]...)
    Ai_v = fill(0.5, 2*(nx-1))

    push!(Ai_i, 1, nx+1)
    push!(Ai_j, 1, nx)
    push!(Ai_v, 1, 1)

    Ai = sparse(Ai_i, Ai_j, Ai_v, nx+1, nx)
    Ii = sparse(I, nx, nx)

    A = [kron(Ii, Ai); kron(Ai, Ii)]
    return A

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

        nx = Int(round(xmax / Δx))
        nt = Int(round(tmax / Δt))

        xs = LinRange(0.5Δx, xmax-0.5Δx, nx)
        ts = LinRange(0, tmax, nt+1)

        cxs = repeat(xs, outer=nx)
        cys = repeat(xs, inner=nx)

        ∇h = build_∇h(nx, Δx)
        A = build_A(nx)

        return new(xs, ts, cxs, cys, Δx, Δt, nx, nt, ∇h, A, ϕ, μ, c, u0)

    end

end