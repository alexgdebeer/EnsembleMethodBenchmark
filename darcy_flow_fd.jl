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

function in_corner(x, y, xmin, xmax, ymin, ymax)
    return x ∈ [xmin, xmax] && y ∈ [ymin, ymax]
end

function get_corner(x, y, xmin, xmax, ymin, ymax)

    (x, y) == (xmin, ymin) && return :bl
    (x, y) == (xmax, ymin) && return :br
    (x, y) == (xmin, ymax) && return :tl
    (x, y) == (xmax, ymax) && return :tr

    error("Point ($x, $y) is not a corner point.")

end

function on_boundary(x, y, xmin, xmax, ymin, ymax)
    return x ∈ [xmin, xmax] || y ∈ [ymin, ymax]
end

function get_boundary(x, y, xmin, xmax, ymin, ymax, bcs)

    x == xmin && return bcs[:x0]
    x == xmax && return bcs[:x1]
    y == ymin && return bcs[:y0]
    y == ymax && return bcs[:y1]

    error("Point ($x, $y) is not on a boundary.")

end

function add_corner_point!(b, rs, cs, vs, c, bcs, i, x, y, Δx, Δy, n_xs, p)

    corner_bnds = Dict(
        :bl => (bcs[:y0], bcs[:x0]), 
        :br => (bcs[:y0], bcs[:x1]),
        :tl => (bcs[:y1], bcs[:x0]),
        :tr => (bcs[:y1], bcs[:x1])
    )

    # Check for a Dirichlet boundary 
    if corner_bnds[c][1].type == :dirichlet
        
        b[i] = corner_bnds[c][1].func(x, y)
        push!(rs, i)
        push!(cs, i)
        push!(vs, 1.0)
        return
        
    elseif corner_bnds[c][2].type == :dirichlet 

        b[i] = corner_bnds[c][2].func(x, y)
        push!(rs, i)
        push!(cs, i)
        push!(vs, 1.0)
        return

    end

    # Apply the Neumann equations
    if c == :bl
        
        b[i] = -(p(x+Δx, y) - 3p(x, y)) * bcs[:x0].func(x, y) / Δx +
               -(p(x, y+Δy) - 3p(x, y)) * bcs[:y0].func(x, y) / Δy

        push!(rs, i, i, i)
        push!(cs, i, i+1, i+n_xs)
        push!(vs, -2p(x, y) / Δx^2 - 2p(x, y) / Δy^2,
                  2p(x, y) / Δx^2,
                  2p(x, y) / Δy^2)
    
    elseif c == :br
        
        b[i] = -(3p(x, y) - p(x-Δx, y)) * bcs[:x1].func(x, y) / Δx +
               -(p(x, y+Δy) - 3p(x, y)) * bcs[:y0].func(x, y) / Δy

        push!(rs, i, i, i)
        push!(cs, i, i-1, i+n_xs)
        push!(vs, -2p(x, y) / Δx^2 - 2p(x, y) / Δy^2,
                  2p(x, y) / Δx^2,
                  2p(x, y) / Δy^2)
    
    elseif c == :tl
        
        b[i] = -(p(x+Δx, y) - 3p(x, y)) * bcs[:x0].func(x, y) / Δx +
               -(3p(x, y) - p(x, y-Δy)) * bcs[:y1].func(x, y) / Δy

        push!(rs, i, i, i)
        push!(cs, i, i+1, i-n_xs)
        push!(vs, -2p(x, y) / Δx^2 - 2p(x, y) / Δy^2,
                  2p(x, y) / Δx^2,
                  2p(x, y) / Δy^2)

    elseif c == :tr
        
        b[i] = -(3p(x, y) - p(x-Δx, y)) * bcs[:x1].func(x, y) / Δx +
               -(3p(x, y) - p(x, y-Δy)) * bcs[:y1].func(x, y) / Δy

        push!(rs, i, i, i)
        push!(cs, i, i-1, i-n_xs)
        push!(vs, -2p(x, y) / Δx^2 - 2p(x, y) / Δy^2,
                  2p(x, y) / Δx^2,
                  2p(x, y) / Δy^2)
    
    end

end

function add_boundary_point!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, n_xs, p)

    if bc.type == :dirichlet 
        add_dirichlet_point!(b, rs, cs, vs, bc, i, x, y)
    elseif bc.type == :neumann 
        add_neumann_point!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, n_xs, p)
    end

end

function add_dirichlet_point!(b, rs, cs, vs, bc, i, x, y)

    b[i] = bc.func(x, y)
    push!(rs, i)
    push!(cs, i)
    push!(vs, 1.0)

end

function add_neumann_point!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, n_xs, p)

    # Add the constant part of the equation
    if bc.name == :y0 
        b[i] = -(p(x, y+Δy) - 3p(x, y)) * bc.func(x, y) / Δy
    elseif bc.name == :y1 
        b[i] = -(3p(x, y) - p(x, y-Δy)) * bc.func(x, y) / Δy
    elseif bc.name == :x0 
        b[i] = -(p(x+Δx, y) - 3p(x, y)) * bc.func(x, y) / Δx
    elseif bc.name == :x1 
        b[i] = -(3p(x, y) - p(x-Δx, y)) * bc.func(x, y) / Δx
    end

    push!(rs, i, i, i, i)

    push!(cs, i)
    push!(vs, -2p(x, y)/Δx^2 - 2p(x, y)/Δy^2)

    # Add the coefficents of components along the x direction
    if bc.name == :x0 
        push!(cs, i+1)
        push!(vs, 2p(x+Δx, y) / Δx^2)
    elseif bc.name == :x1 
        push!(cs, i-1)
        push!(vs, 2p(x-Δx, y) / Δx^2)
    else 
        push!(cs, i+1, i-1)
        push!(vs, (0.25p(x+Δx, y) - 0.25p(x-Δx, y) + p(x, y)) / Δx^2,
                  (0.25p(x-Δx, y) - 0.25p(x+Δx, y) + p(x, y)) / Δx^2)
    end

    # Add the coefficents of components along the y direction
    if bc.name == :y0 
        push!(cs, i+n_xs)
        push!(vs, 2p(x, y+Δy) / Δy^2)
    elseif bc.name == :y1 
        push!(cs, i-n_xs)
        push!(vs, 2p(x, y-Δy) / Δy^2)
    else 
        push!(cs, i+n_xs, i-n_xs)
        push!(vs, (0.25p(x, y+Δy) - 0.25p(x, y-Δy) + p(x, y)) / Δy^2,
                  (0.25p(x, y-Δy) - 0.25p(x, y+Δy) + p(x, y)) / Δy^2)
    end

end

function add_interior_point!(rs, cs, vs, i, x, y, Δx, Δy, n_xs, p)

    push!(rs, i, i, i, i, i)
    push!(cs, i, i+1, i-1, i+n_xs, i-n_xs)
    
    push!(
        vs,
        -2p(x, y) / Δx^2 - 2p(x, y) / Δy^2,
        (0.25p(x+Δx, y) - 0.25p(x-Δx, y) + p(x, y)) / Δx^2,
        (0.25p(x-Δx, y) - 0.25p(x+Δx, y) + p(x, y)) / Δx^2,
        (0.25p(x, y+Δy) - 0.25p(x, y-Δy) + p(x, y)) / Δy^2,
        (0.25p(x, y-Δy) - 0.25p(x, y+Δy) + p(x, y)) / Δy^2
    )

end

function generate_grid(xs, ys, Δx, Δy, p, bcs)

    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)

    n_xs = length(xs)
    n_ys = length(ys)
    n_us = n_xs * n_ys

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

        if in_corner(x, y, xmin, xmax, ymin, ymax)

            c = get_corner(x, y, xmin, xmax, ymin, ymax)
            add_corner_point!(b, rs, cs, vs, c, bcs, i, x, y, Δx, Δy, n_xs, p)

        elseif on_boundary(x, y, xmin, xmax, ymin, ymax)

            bc = get_boundary(x, y, xmin, xmax, ymin, ymax, bcs)
            add_boundary_point!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, n_xs, p)
        
        else
        
            add_interior_point!(rs, cs, vs, i, x, y, Δx, Δy, n_xs, p)
        
        end

    end

    A = sparse(rs, cs, vs, n_us, n_us)

    return A, b

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
function p(x, y)
    if x > xmax || x < xmin || y > ymax || y < ymin
        error("$x, $y")
    end
    return 2.0 #+ 0.02rand()
end

# Set up boundary conditions
x0 = BoundaryCondition(:x0, :neumann, (x, y) -> 0.0)
x1 = BoundaryCondition(:x1, :neumann, (x, y) -> 0.0)
y0 = BoundaryCondition(:y0, :dirichlet, (x, y) -> 0.0)
y1 = BoundaryCondition(:y1, :neumann, (x, y) -> 2.0)

# Define a mapping between boundary condition names and objects
bcs = Dict(:y0 => y0, :y1 => y1, :x0 => x0, :x1 => x1)

A, b = generate_grid(xs, ys, Δx, Δy, p, bcs)

prob = LinearProblem(A, b)
sol = solve(prob)
u = reshape(sol.u, n_xs, n_ys)

heatmap(xs, ys, rotr90(u))