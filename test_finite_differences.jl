using LinearAlgebra

Δx = 1.0
nx = 10

∇hi_i = repeat(2:(nx-1), inner=2)
∇hi_j = vcat([[i, i+1] for i ∈ 2:(nx-1)]...)
∇hi_v = repeat([-1.0, 1.0], outer=(nx-2))

push!(∇hi_i, 1, 1, 1, nx, nx, nx)
push!(∇hi_j, 1, 2, 3, nx-2, nx-1, nx)
push!(∇hi_v, -3/2, 2, -1/2, 1/2, -2, 3/2)

∇hi = sparse(∇hi_i, ∇hi_j, ∇hi_v, nx, nx) / Δx
Ii = sparse(I, nx, nx)

∇h = [kron(Ii, ∇hi); kron(∇hi, Ii)]