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
    nx::Real 
    nt::Real 

    ∇h::SparseMatrixCSC
    A::SparseMatrixCSC

    ϕ::Real 
    μ::Real
    c::Real

    function Grid(
        Δx::Real,
        Δt::Real,
        nx::Int,
        nt::Int; 
        ϕ::Real=1.0,
        μ::Real=1.0,
        c::Real=1.0
    )

        ∇h = build_∇h(nx, Δx)
        A = build_A(nx)

        return new(Δx, Δt, nx, nt, ∇h, A, ϕ, μ, c)

    end

end

function build_fd_matrices(nx, Δ)

    # Short central difference operator

    # Inner points
    D_i = repeat(2:nx, inner=2)
    D_j = vcat([[i-1, i] for i ∈ 2:nx]...)
    D_v = repeat([-1, 1], outer=(nx-1))

    # Neumann boundaries (TODO: check these...)
    push!(D_i, 1, 1, nx+1, nx+1)
    push!(D_j, 1, 2, nx-1, nx)
    push!(D_v, -1, 1, -1, 1)

    D = sparse(D_i, D_j, D_v, nx+1, nx) / Δ

    # Interpolation operator
    W_i = repeat(2:nx, inner=2)
    W_j = vcat([[i-1, i] for i ∈ 2:nx]...)
    W_v = fill(0.5, 2*(nx-1))

    push!(W_i, 1, nx+1)
    push!(W_j, 1, nx)
    push!(W_v, 1, 1)

    W = sparse(W_i, W_j, W_v, nx+1, nx)

    Id = sparse(I, nx, nx)
    ∇h = [kron(Id, D); kron(D, Id)]
    A = [kron(Id, W); kron(W, Id)]

    return ∇h, A

end

Δx = 0.1
Δt = 1

nx = 10
nt = 10

g = Grid(Δx, Δt, nx, nt);