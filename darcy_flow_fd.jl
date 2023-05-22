using LinearAlgebra
using LinearSolve
using Plots
using Random
using SparseArrays

Random.seed!(16)

struct BC
    type::Symbol
    func::Function
end

# Define the grid dimensions
xmin, Δx, xmax = 0.0, 0.01, 1.0
ymin, Δy, ymax = 0.0, 0.01, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

n_xs = length(xs)
n_ys = length(ys)
n_us = n_xs*n_ys

# TODO: define permeability interpolation object
p(x, y) = 2.0

# Set up boundary conditions
# x0 = BC(:dirichlet, (x, y) -> y/100.0)
# x1 = BC(:dirichlet, (x, y) -> y/100.0)
# y0 = BC(:dirichlet, (x, y) -> 0.0)
# y1 = BC(:dirichlet, (x, y) -> 1.0)
x0 = BC(:neumann, (x, y) -> 0.0)
x1 = BC(:neumann, (x, y) -> 0.0)
y0 = BC(:dirichlet, (x, y) -> 0.0)
y1 = BC(:neumann, (x, y) -> -2.0)

function generate_grid(xs, ys, Δx, Δy, p, x0, x1, y0, y1)

    # TODO: make b sparse?
    b = zeros(n_us)

    rows = Int64[]
    cols = Int64[]
    vals = Float64[]

    for i ∈ 1:n_us 

        # Find the x and y coordinates of the current point
        x = xs[(i-1)%n_xs+1] 
        y = ys[Int(ceil(i/n_xs))]

        # Check for corner point
        if [x, y] ∈ [[xs[1], ys[1]], [xs[1], ys[end]], [xs[end], ys[1]], [xs[end], ys[end]]]

            push!(rows, i)
            push!(cols, i)
            push!(vals, 1.0)

        elseif x ∈ [xs[1], xs[end]] || y ∈ [ys[1], ys[end]]

            # Bottom boundary
            if y == ys[1]
                
                if y0.type == :dirichlet

                    b[i] = y0.func(x, y)

                    push!(rows, i)
                    push!(cols, i)
                    push!(vals, 1.0)

                elseif y0.type == :neumann 
                    
                    b[i] = ((p(x, y+Δy) - p(x, y)) * y0.func(x, y) - 2p(x, y) * y0.func(x, y)) / Δy

                    push!(rows, i, i, i, i)
                    push!(cols, i, i+1, i-1, i+n_xs)
                    push!(
                        vals,
                        -2p(x, y)/Δx^2 - 2p(x, y)/Δy^2,
                        (0.25p(x+Δx, y) - 0.25p(x-Δx, y) + p(x, y))/(Δx^2),
                        (0.25p(x-Δx, y) - 0.25p(x+Δx, y) + p(x, y))/(Δx^2),
                        2p(x, y+Δy)/(Δy^2)
                    )

                end

            # Top boundary
            elseif y == ys[end]

                if y1.type == :dirichlet

                    b[i] = y1.func(x, y)

                    push!(rows, i)
                    push!(cols, i)
                    push!(vals, 1.0)

                elseif y1.type == :neumann 

                    b[i] = ((p(x, y) - p(x, y-Δy)) * y1.func(x, y) + 2p(x, y) * y1.func(x, y)) / Δy

                    push!(rows, i, i, i, i)
                    push!(cols, i, i+1, i-1, i-n_xs)
                    push!(
                        vals,
                        -2p(x, y)/Δx^2 - 2p(x, y)/Δy^2,
                        (0.25p(x+Δx, y) - 0.25p(x-Δx, y) + p(x, y))/(Δx^2),
                        (0.25p(x-Δx, y) - 0.25p(x+Δx, y) + p(x, y))/(Δx^2),
                        2p(x, y-Δy)/(Δy^2)
                    )

                end

            # Left hand boundary 
            elseif x == xs[1] 
                
                if x0.type == :dirichlet

                    b[i] = x0.func(x, y)

                    push!(rows, i)
                    push!(cols, i)
                    push!(vals, 1.0)

                elseif x0.type == :neumann 
                    
                    b[i] = ((p(x+Δx, y) - p(x, y)) * x0.func(x, y) - 2p(x, y) * x0.func(x, y)) / Δx

                    push!(rows, i, i, i, i)
                    push!(cols, i, i+1, i+n_xs, i-n_xs)
                    push!(
                        vals,
                        -2p(x, y)/Δx^2 - 2p(x, y)/Δy^2,
                        2p(x+Δx, y)/(Δx^2),
                        (0.25p(x, y+Δy) - 0.25p(x, y-Δy) + p(x, y))/(Δy^2),
                        (0.25p(x, y-Δy) - 0.25p(x, y+Δy) + p(x, y))/(Δy^2)
                    )

                end

            # Right hand boundary
            elseif x == xs[end]
                
                if x1.type == :dirichlet

                    b[i] = x1.func(x, y)

                    push!(rows, i)
                    push!(cols, i)
                    push!(vals, 1.0)

                elseif x1.type == :neumann 

                    b[i] = ((p(x, y) - p(x-Δx, y)) * x1.func(x, y) + 2p(x, y) * x1.func(x, y)) / Δx

                    push!(rows, i, i, i, i)
                    push!(cols, i, i-1, i+n_xs, i-n_xs)
                    push!(
                        vals,
                        -2p(x, y)/Δx^2 - 2p(x, y)/Δy^2,
                        2p(x-Δx, y)/(Δx^2),
                        (0.25p(x, y+Δy) - 0.25p(x, y-Δy) + p(x, y))/(Δy^2),
                        (0.25p(x, y-Δy) - 0.25p(x, y+Δy) + p(x, y))/(Δy^2)
                    )

                    # error("Not implemented yet.")
                end

            end

        else

            # Interior point
            push!(rows, i, i, i, i, i)
            push!(cols, i, i+1, i-1, i+n_xs, i-n_xs)
            push!(
                vals,
                -(2p(x,y))/(Δx^2) - (2p(x,y))/(Δy^2),
                (0.25p(x+Δx, y) - 0.25p(x-Δx, y) + p(x, y))/(Δx^2),
                (0.25p(x-Δx, y) - 0.25p(x+Δx, y) + p(x, y))/(Δx^2),
                (0.25p(x, y+Δy) - 0.25p(x, y-Δy) + p(x, y))/(Δy^2),
                (0.25p(x, y-Δy) - 0.25p(x, y+Δy) + p(x, y))/(Δy^2)
            )

        end

    end

    A = sparse(rows, cols, vals, n_us, n_us)

    return A, b

end

A, b = @time generate_grid(xs, ys, Δx, Δy, p, x0, x1, y0, y1)

prob = LinearProblem(A, b)
sol = solve(prob)
u = reshape(sol.u, n_xs, n_ys)
heatmap(xs, ys, u)