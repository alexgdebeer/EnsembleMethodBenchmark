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

    Δx::Real 
    Δt::Real

    nx::Int
    nt::Int 

    ∇h::SparseMatrixCSC
    A::SparseMatrixCSC

    ϕ::Real 
    μ::Real
    c::Real
    u0::Real

    function Grid(
        Δx::Real,
        Δt::Real,
        nx::Int,
        nt::Int; 
        ϕ::Real=1.0,
        μ::Real=1.0,
        c::Real=1.0,
        u0::Real=1.0
    )

        ∇h = build_∇h(nx, Δx)
        A = build_A(nx)

        return new(Δx, Δt, nx, nt, ∇h, A, ϕ, μ, c, u0)

    end

end

Δx = 0.1
Δt = 1

nx = 10
nt = 10

g = Grid(Δx, Δt, nx, nt);