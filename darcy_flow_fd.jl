using LinearAlgebra
using LinearSolve
using Plots
using Random
using SparseArrays

#Random.seed!(16)

struct BoundaryCondition
    name::Symbol
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
p(x, y) = 2.0+0.01rand()

# Set up boundary conditions
x0 = BoundaryCondition(:x0, :neumann, (x, y) -> 0.0)
x1 = BoundaryCondition(:x1, :neumann, (x, y) -> 0.0)
y0 = BoundaryCondition(:y0, :dirichlet, (x, y) -> 0.0)
y1 = BoundaryCondition(:y1, :neumann, (x, y) -> 2.0)

# Define a mapping between boundary condition names and objects
bcs = Dict(:y0 => y0, :y1 => y1, :x0 => x0, :x1 => x1)

function on_boundary(x, y, xmin, xmax, ymin, ymax)
    return x ∈ [xmin, xmax] || y ∈ [ymin, ymax]
end

function get_boundary(x, y, xmin, xmax, ymin, ymax, bcs)
    
    if x == xmin
        return bcs[:x0]
    elseif x == xmax
        return bcs[:x1]
    elseif y == ymin
        return bcs[:y0]
    elseif y == ymax
        return bcs[:y1]
    end

end

function update_boundary!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, p)

    if bc.type == :dirichlet 
        update_dirichlet!(b, rs, cs, vs, bc, i, x, y)
    elseif bc.type == :neumann 
        update_neumann!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, p)
    end

end

function update_dirichlet!(b, rs, cs, vs, bc, i, x, y)

    b[i] = bc.func(x, y)
    push!(rs, i)
    push!(cs, i)
    push!(vs, 1.0)

end

function update_neumann!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, p)

    # Add the constant part of the equation
    if bc.name == :y0
        b[i] = -((p(x, y+Δy) - p(x, y)) * y0.func(x, y) - 2p(x, y) * y0.func(x, y)) / Δy
    elseif bc.name == :y1
        b[i] = -((p(x, y) - p(x, y-Δy)) * y1.func(x, y) + 2p(x, y) * y1.func(x, y)) / Δy
    elseif bc.name == :x0 
        b[i] = -((p(x+Δx, y) - p(x, y)) * x0.func(x, y) - 2p(x, y) * x0.func(x, y)) / Δx
    elseif bc.name == :x1 
        b[i] = -((p(x, y) - p(x-Δx, y)) * x1.func(x, y) + 2p(x, y) * x1.func(x, y)) / Δx
    end

    push!(rs, i, i, i, i)

    push!(cs, i)
    push!(vs, -2p(x, y)/Δx^2 - 2p(x, y)/Δy^2)

    # Add the coefficents of components along the x direction
    if bc.name == :x0 
        push!(cs, i+1)
        push!(vs, 2p(x+Δx, y)/Δx^2)
    elseif bc.name == :x1 
        push!(cs, i-1)
        push!(vs, 2p(x-Δx, y)/Δx^2)
    else 
        push!(cs, i+1, i-1)
        push!(
            vs, 
            (0.25p(x+Δx, y) - 0.25p(x-Δx, y) + p(x, y))/Δx^2,
            (0.25p(x-Δx, y) - 0.25p(x+Δx, y) + p(x, y))/Δx^2
        )
    end

    # Add the coefficents of components along the y direction
    if bc.name == :y0 
        push!(cs, i+n_xs)
        push!(vs, 2p(x, y+Δy)/Δy^2)
    elseif bc.name == :y1 
        push!(cs, i-n_xs)
        push!(vs, 2p(x, y-Δy)/Δy^2)
    else 
        push!(cs, i+n_xs, i-n_xs)
        push!(
            vs, 
            (0.25p(x, y+Δy) - 0.25p(x, y-Δy) + p(x, y))/Δy^2,
            (0.25p(x, y-Δy) - 0.25p(x, y+Δy) + p(x, y))/Δy^2
        )
    end

end

function update_interior_point!(rs, cs, vs, i, x, y, Δx, Δy, p)

    push!(rs, i, i, i, i, i)
    push!(cs, i, i+1, i-1, i+n_xs, i-n_xs)
    
    push!(
        vs,
        -(2p(x,y))/(Δx^2) - (2p(x,y))/(Δy^2),
        (0.25p(x+Δx, y) - 0.25p(x-Δx, y) + p(x, y))/(Δx^2),
        (0.25p(x-Δx, y) - 0.25p(x+Δx, y) + p(x, y))/(Δx^2),
        (0.25p(x, y+Δy) - 0.25p(x, y-Δy) + p(x, y))/(Δy^2),
        (0.25p(x, y-Δy) - 0.25p(x, y+Δy) + p(x, y))/(Δy^2)
    )

end

function generate_grid(xs, ys, Δx, Δy, p, bcs)

    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)

    corner_points = Set([(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)])

    # Initalise components of A
    rs = Int64[]
    cs = Int64[]
    vs = Float64[]

    # TODO: make b sparse?
    b = zeros(n_us)

    for i ∈ 1:n_us 

        # Find the x and y coordinates of the current point
        x = xs[(i-1)%n_xs+1] 
        y = ys[Int(ceil(i/n_xs))]

        if (x, y) ∈ corner_points

            # TODO: fix this up
            push!(rs, i)
            push!(cs, i)
            push!(vs, 1.0)

        elseif on_boundary(x, y, xmin, xmax, ymin, ymax)

            bc = get_boundary(x, y, xmin, xmax, ymin, ymax, bcs)
            update_boundary!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, p)
        
        else
        
            update_interior_point!(rs, cs, vs, i, x, y, Δx, Δy, p)
        
        end

    end

    A = sparse(rs, cs, vs, n_us, n_us)

    return A, b

end

A, b = @time generate_grid(xs, ys, Δx, Δy, p, bcs)

prob = LinearProblem(A, b)
sol = solve(prob)
u = reshape(sol.u, n_xs, n_ys)
heatmap(xs, ys, u)