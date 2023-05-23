using Interpolations
using LinearAlgebra
using LinearSolve
using SparseArrays

struct BoundaryCondition
    name::Symbol
    type::Symbol
    func::Function
end

function in_corner(x, y, xmin, xmax, ymin, ymax)
    return x ∈ [xmin, xmax] && y ∈ [ymin, ymax]
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

function add_corner_point!(rs, cs, vs, i)

    push!(rs, i)
    push!(cs, i)
    push!(vs, 1.0)

end

function add_boundary_point!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, n_xs)

    if bc.type == :dirichlet 
        add_dirichlet_point!(b, rs, cs, vs, bc, i, x, y)
    elseif bc.type == :neumann 
        add_neumann_point!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, n_xs)
    end

end

function add_dirichlet_point!(b, rs, cs, vs, bc, i, x, y)

    b[i] = bc.func(x, y)
    push!(rs, i)
    push!(cs, i)
    push!(vs, 1.0)

end

function add_neumann_point!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, n_xs)

    b[i] = bc.func(x, y)

    push!(rs, i, i, i)

    if bc.name == :y0 
        push!(cs, i, i+n_xs, i+2n_xs)
        push!(vs, -3.0 / 2Δy, 4.0 / 2Δy, -1.0 / 2Δy)
    elseif bc.name == :y1 
        push!(cs, i, i-n_xs, i-2n_xs)
        push!(vs, 3.0 / 2Δy, -4.0 / 2Δy, 1.0 / 2Δy)
    elseif bc.name == :x0 
        push!(cs, i, i+1, i+2)
        push!(vs, -3.0 / 2Δx, 4.0 / 2Δx, -1.0 / 2Δx)
    elseif bc.name == :x1 
        push!(cs, i, i-1, i-2)
        push!(vs, 3.0 / 2Δx, -4.0 / 2Δx, 1.0 / 2Δx)
    end

end

function add_interior_point!(rs, cs, vs, i, x, y, Δx, Δy, n_xs, p)

    push!(rs, i, i, i, i, i)
    push!(cs, i, i+1, i-1, i+n_xs, i-n_xs)
    
    push!(vs, -2p(x, y) / Δx^2 - 2p(x, y) / Δy^2,
              (0.25p(x+Δx, y) - 0.25p(x-Δx, y) + p(x, y)) / Δx^2,
              (0.25p(x-Δx, y) - 0.25p(x+Δx, y) + p(x, y)) / Δx^2,
              (0.25p(x, y+Δy) - 0.25p(x, y-Δy) + p(x, y)) / Δy^2,
              (0.25p(x, y-Δy) - 0.25p(x, y+Δy) + p(x, y)) / Δy^2)

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
    
    b = zeros(n_us)

    for i ∈ 1:n_us 

        # Find the x and y coordinates of the current point
        x = xs[(i-1)%n_xs+1] 
        y = ys[Int(ceil(i/n_xs))]

        if in_corner(x, y, xmin, xmax, ymin, ymax)

            add_corner_point!(rs, cs, vs, i)

        elseif on_boundary(x, y, xmin, xmax, ymin, ymax)

            bc = get_boundary(x, y, xmin, xmax, ymin, ymax, bcs)
            add_boundary_point!(b, rs, cs, vs, bc, i, x, y, Δx, Δy, n_xs)
        
        else
        
            add_interior_point!(rs, cs, vs, i, x, y, Δx, Δy, n_xs, p)
        
        end

    end

    A = sparse(rs, cs, vs, n_us, n_us)

    return A, b

end

using Distributions
using Random

function exp_squared_cov(σ, γ, xs, ys)

    # Generate vectors of x and y coordinates
    cxs = vec([x for _ ∈ ys, x ∈ xs])
    cys = vec([y for y ∈ ys, _ ∈ xs])

    # Generate a matrix of distances between each set of coordinates
    ds = (cxs .- cxs').^2 + (cys .- cys').^2

    Γ = σ^2 * exp.(-(1/2γ^2).*ds) + 1e-6I

    return Γ

end

function sample_perms(d) 
    return exp.(reshape(rand(d), length(xs), length(ys)))
end