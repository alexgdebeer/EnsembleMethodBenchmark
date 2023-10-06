using SparseArrays
using LinearAlgebra

# Define cell centres
xmin, xmax = 0, 10
Δ = 1.0

xs = (xmin+0.5Δ):Δ:(xmax-0.5Δ) 

nx = length(xs)
nu = nx^2                   # Number of cell centres
nf = (nx+1)^2               # Number of faces

# Form short central difference operator (assuming Neumann points end 
# up the same as all other points)
is = repeat(1:nx, inner=2)
js = vcat([[i, i+1] for i ∈ 1:nx]...)
vs = repeat([-1, 1], outer=nx)

D = sparse(is, js, vs, nx, nx+1) / Δ # Row for every centre, column for every face (in 1d)
Id = sparse(I, nx, nx)

# Gradient operator (TODO: check for correctness)
∇h = [kron(Id, D); kron(D, Id)]

# Weighting matrix
# rows = cell faces 
# columns = cell centres 

# Each row should be the 