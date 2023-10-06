using SparseArrays
using LinearAlgebra

include("darcy_flow/setup/setup.jl")

mf = MaternField(grid_f, logp_mu, σ_bounds, l_bounds)
θs = rand(mf)
us = transform(mf, vec(θs))

# Define cell centres
xmin, xmax = 0, 10
Δ = 1.0

xs = xmin:Δ:xmax 

nx = length(xs)
nu = nx^2

is = Int[]
js = Int[]
vs = Float64[]

# Neumann points
push!(is, 1, 1, nx, nx)
push!(js, 1, 2, nx-1, nx)
push!(vs, -1, 1, -1, 1)

# Inner points
for i ∈ 2:(nx-1)
    push!(is, i, i)
    push!(js, i-1, i+1)
    push!(vs, -0.5, 0.5)
end

D = sparse(is, js, vs, nx, nx) / Δ
Id = sparse(I, nx, nx)

# Gradient operator (TODO: check for correctness)
∇h = [kron(Id, D); kron(D, Id)]