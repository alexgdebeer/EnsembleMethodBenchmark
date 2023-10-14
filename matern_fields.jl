using Distributions
using LinearAlgebra
using LinearSolve
using SparseArrays
using SpecialFunctions: gamma

GRAD_2D = [-1.0 1.0 0.0; 
           -1.0 0.0 1.0]

xmin = 1
xmax = 10
Δx = 0.1
xs = xmin:Δx:xmax

nx = length(xs)

elements = []
boundary_elements = []

g = reshape(1:nx^2, nx, nx)

for j ∈ 1:(nx-1), i ∈ 1:(nx-1)

    push!(elements, [g[i, j], g[i+1, j], g[i+1, j+1]])
    push!(elements, [g[i, j], g[i, j+1], g[i+1, j+1]])

end

points = hcat([[x, y] for x ∈ xs for y ∈ xs]...)
elements = hcat(elements...)

facets_x0 = hcat([[(i-1)*nx+1, i*nx+1] for i ∈ 1:(nx-1)]...)
facets_x1 = hcat([[i*nx, (i+1)*nx] for i ∈ 1:(nx-1)]...)
facets_y0 = hcat([[i, i+1] for i ∈ 1:(nx-1)]...)
facets_y1 = hcat([[i, i+1] for i ∈ (nx^2-nx+1):(nx^2-1)]...)

boundary_facets = hcat(facets_x0, facets_x1, facets_y0, facets_y1)

n_points = nx^2
n_elements = 2(nx-1)^2
n_boundary_facets = 4*(nx-1)

M_i, M_j, M_v = Int[], Int[], Float64[]
K_i, K_j, K_v = Int[], Int[], Float64[]
N_i, N_j, N_v = Int[], Int[], Float64[]

for e ∈ eachcol(elements)

    for i ∈ 1:3

        # TODO: can probably make specific to triangles
        T = hcat(points[:, e[i%3+1]] - points[:, e[i]],
                 points[:, e[(i+1)%3+1]] - points[:, e[i]])

        detT = abs(det(T))
        invT = inv(T)

        for j ∈ 1:3

            push!(M_i, e[i])
            push!(M_j, e[j])
            i == j && push!(M_v, detT/12)
            i != j && push!(M_v, detT/24)

            push!(K_i, e[i])
            push!(K_j, e[j])
            push!(K_v, 1/2 * detT * GRAD_2D[:, 1]' * invT * invT' * GRAD_2D[:, (j-i+3)%3+1])

        end

    end

end

for (fi, fj) ∈ eachcol(boundary_facets)

    det = norm(points[:, fi] - points[:, fj])

    push!(N_i, [fi, fj, fi, fj]...)
    push!(N_j, [fi, fj, fj, fi]...)
    push!(N_v, [det * 1/3, det * 1/3, det * 1/6, det * 1/6]...)

end

M = sparse(M_i, M_j, M_v, nx^2, nx^2)
K = sparse(K_i, K_j, K_v, nx^2, nx^2)
N = sparse(N_i, N_j, N_v, nx^2, nx^2)
L = sparse(cholesky(Hermitian(M)).L)

σ = 1.0
l = 2.0
ν = 1
λ = 1.42 * l

α = σ^2 * (4 * pi * gamma(ν+1)) / gamma(ν)

W = rand(Normal(), nx^2)

A = M + K + l^2 / λ * N
b = √(α * l^2) * L * W

x = solve(LinearProblem(A, b))
