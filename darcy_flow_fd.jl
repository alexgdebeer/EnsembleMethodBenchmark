using LinearAlgebra
using LinearSolve
using Plots
using SparseArrays

# Define the grid dimensions
xmin, Δx, xmax = 0.0, 1.0, 100.0
ymin, Δy, ymax = 0.0, 1.0, 100.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

n_xs = length(xs)
n_ys = length(ys)
n_us = n_xs*n_ys

# TODO: define permeability interpolation object
p(x, y) = 0.01rand()+0.5

# Dirichlet boundary conditions
x0(y) = y/100.0
x1(y) = y/100.0

y0(x) = 0.0
y1(x) = 1.0

# TODO: clean up b
b = zeros(n_us)

rows = Int64[]
cols = Int64[]
vals = Float64[]

for i ∈ 1:n_us 

    # Find the x and y coordinates of the current point
    x = xs[(i-1)%n_xs+1] 
    y = ys[Int(ceil(i/n_xs))]

    if x ∈ [xs[1], xs[end]] || y ∈ [ys[1], ys[end]]

        push!(rows, i)
        push!(cols, i)
        push!(vals, 1.0)

        # Bottom boundary
        if x == xs[1]
            b[i] = x0(y)

        # Top boundary
        elseif x == xs[end]
            b[i] = x1(y)

        # Left hand boundary 
        elseif y == ys[1] 
            b[i] = y0(x)

        # Right hand boundary
        elseif y == ys[end]
            b[i] = y1(x)

        end

    else

        # Fill in the stencil
        push!(rows, i, i, i, i, i)
        push!(cols, i, i+n_xs, i-n_xs, i+1, i-1)
        
        # push!(
        #     vals,
        #     -(p(x+Δx/2, y) + p(x-Δx/2, y))/(Δx^2) - (p(x, y+Δy/2) + p(x, y-Δy/2))/(Δy^2),
        #     p(x, y+Δy/2)/(Δy^2),
        #     p(x, y-Δy/2)/(Δy^2),
        #     p(x+Δx/2, y)/(Δx^2),
        #     p(x-Δx/2, y)/(Δx^2)
        # )
        push!(
            vals,
            -(2p(x,y))/(Δx^2) - (2p(x,y))/(Δy^2),
            (0.25p(x, y+Δy) - 0.25p(x, y-Δy) + p(x, y))/(Δy^2),
            (0.25p(x, y-Δy) - 0.25p(x, y+Δy) + p(x, y))/(Δy^2),
            (0.25p(x+Δx, y) - 0.25p(x-Δx, y) + p(x, y))/(Δx^2),
            (0.25p(x-Δx, y) - 0.25p(x+Δx, y) + p(x, y))/(Δx^2)
        )

    end

end

A = sparse(rows, cols, vals, n_us, n_us)

prob = LinearProblem(A, b)
@time sol = solve(prob)
u = reshape(sol.u, n_xs, n_ys)
heatmap(xs, ys, u)