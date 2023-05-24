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

    push!(
        vs, 
        -(p(x+0.5Δx, y) + p(x-0.5Δx, y)) / Δx^2 - 
            (p(x, y+0.5Δy) + p(x, y-0.5Δy)) / Δy^2,
        p(x+0.5Δx, y) / Δx^2,
        p(x-0.5Δx, y) / Δx^2,
        p(x, y+0.5Δy) / Δy^2,
        p(x, y-0.5Δy) / Δy^2
    )

end

function generate_grid(xs, ys, p, bcs)

    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)

    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]

    n_xs = length(xs)
    n_ys = length(ys)
    n_us = n_xs * n_ys

    # Initalise components of A
    rs = Int[]
    cs = Int[]
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