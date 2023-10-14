using Distributions
using LinearAlgebra
using LinearSolve
using SparseArrays
using SpecialFunctions: gamma
# TODO: check Python code -- why is there no l^2 in the A equation?

GRAD_2D = [-1.0 1.0 0.0; 
           -1.0 0.0 1.0]

xmin = 0
xmax = 10
Δx = 0.5
xs = xmin:Δx:xmax

nx = length(xs)

elements = []
boundary_elements = []

g = reshape(1:nx^2, nx, nx)

for j ∈ 1:(nx-1), i ∈ 1:(nx-1)

    push!(elements, [g[i, j], g[i+1, j], g[i+1, j+1]])
    push!(elements, [g[i, j], g[i, j+1], g[i+1, j+1]])

end

points = hcat([[x, y] for y ∈ xs for x ∈ xs]...)
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

detT = Δx^2

@time for (n, e) ∈ enumerate(eachcol(elements))

    for i ∈ 1:3

        T = hcat(points[:, e[i%3+1]] - points[:, e[i]],
                 points[:, e[(i+1)%3+1]] - points[:, e[i]])

        invT = inv(T)

        for j ∈ 1:3

            push!(M_i, e[i])
            push!(M_j, e[j])
            i == j && push!(M_v, Δx^2/12)
            i != j && push!(M_v, Δx^2/24)

            push!(K_i, e[i])
            push!(K_j, e[j])
            push!(K_v, 1/2 * detT * GRAD_2D[:, 1]' * invT * invT' * GRAD_2D[:, (j-i+3)%3+1])

        end

    end

end

for (fi, fj) ∈ eachcol(boundary_facets)

    push!(N_i, fi, fj, fi, fj)
    push!(N_j, fi, fj, fj, fi)
    push!(N_v, Δx/3, Δx/3, Δx/6, Δx/6)

end

M = sparse(M_i, M_j, M_v, nx^2, nx^2)
K = sparse(K_i, K_j, K_v, nx^2, nx^2)
N = sparse(N_i, N_j, N_v, nx^2, nx^2)
L = sparse(cholesky(Hermitian(M)).L)

σ = 1.0
l = 3
ν = 1.0

α = σ^2 * (4 * pi * gamma(ν+1)) / gamma(ν)

A = M + l^2 * K + l / 1.42 * N

n_samples = 1000
XS = zeros(nx^2, n_samples)

for i ∈ 1:n_samples

    W = rand(Normal(), nx^2)
    XS[:, i] = solve(LinearProblem(A, √(α * l^2) * L * W))

    if i % 100 == 0
        println(i)
    end

end

σs = std(XS, dims=2)
