using LinearAlgebra
using LinearSolve
using Plots

# Define the grid dimensions
xmin, Δx, xmax = 0.0, 1.0, 10.0
ymin, Δy, ymax = 0.0, 1.0, 10.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

n_xs = length(xs)
n_ys = length(ys)
n_us = n_xs*n_ys

# TODO: define permeability interpolation object
p(x, y) = 1.0

# Dirichlet boundary conditions
x0(y) = y/10.0
x1(y) = y/10.0

y0(x) = 0.0
y1(x) = 1.0

A = zeros(n_us, n_us)
b = zeros(n_us)

for i ∈ 1:n_us 

    # Find the x and y coordinates of the current point
    x = xs[(i-1)%n_xs+1] 
    y = ys[Int(ceil(i/n_xs))]

    A[i,i] = 1

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

    else

        # Fill in the stencil
        A[i,i] = -(p(x+Δx/2, y) + p(x-Δx/2, y))/(Δx^2) - (p(x, y+Δy/2) + p(x, y-Δy/2))/(Δy^2)
        
        A[i,i+n_xs] = p(x, y+Δy/2)/(Δy^2)
        A[i,i-n_xs] = p(x, y-Δy/2)/(Δy^2)
        
        A[i,i+1] = p(x+Δx/2, y)/(Δx^2)
        A[i,i-1] = p(x-Δx/2, y)/(Δx^2)

    end

end

prob = LinearProblem(A, b)
sol = solve(prob)
u = reshape(sol.u, n_xs, n_ys)
heatmap(xs, ys, u)