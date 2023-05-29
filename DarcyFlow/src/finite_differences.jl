struct Grid
    xs::AbstractVector
    ys::AbstractVector
    xmin::Real
    xmax::Real
    ymin::Real
    ymax::Real
    Δx::Real
    Δy::Real
    nx::Int
    ny::Int
    nu::Int
end

struct BoundaryCondition
    name::Symbol
    type::Symbol
    func::Function
end

function in_corner(x, y, g)
    return x ∈ [g.xmin, g.xmax] && y ∈ [g.ymin, g.ymax]
end

function on_boundary(x, y, g)
    return x ∈ [g.xmin, g.xmax] || y ∈ [g.ymin, g.ymax]
end

function get_boundary(x, y, g, bcs)

    x == g.xmin && return bcs[:x0]
    x == g.xmax && return bcs[:x1]
    y == g.ymin && return bcs[:y0]
    y == g.ymax && return bcs[:y1]

    error("Point ($x, $y) is not on a boundary.")

end

function add_corner_point!(rs, cs, vs, i)

    push!(rs, i)
    push!(cs, i)
    push!(vs, 1.0)

end

function add_boundary_point!(b, rs, cs, vs, i, x, y, g, bc)

    if bc.type == :dirichlet 
        add_dirichlet_point!(b, rs, cs, vs, i, x, y, bc)
    elseif bc.type == :neumann 
        add_neumann_point!(b, rs, cs, vs, i, x, y, g, bc)
    end

end

function add_dirichlet_point!(b, rs, cs, vs, i, x, y, bc)

    b[i] = bc.func(x, y)
    push!(rs, i)
    push!(cs, i)
    push!(vs, 1.0)

end

function add_neumann_point!(b, rs, cs, vs, i, x, y, g, bc)

    b[i] = bc.func(x, y)

    push!(rs, i, i, i)

    if bc.name == :y0 
        push!(cs, i, i+g.nx, i+2g.nx)
        push!(vs, -3.0 / 2g.Δy, 4.0 / 2g.Δy, -1.0 / 2g.Δy)
    elseif bc.name == :y1 
        push!(cs, i, i-g.nx, i-2g.nx)
        push!(vs, 3.0 / 2g.Δy, -4.0 / 2g.Δy, 1.0 / 2g.Δy)
    elseif bc.name == :x0 
        push!(cs, i, i+1, i+2)
        push!(vs, -3.0 / 2g.Δx, 4.0 / 2g.Δx, -1.0 / 2g.Δx)
    elseif bc.name == :x1 
        push!(cs, i, i-1, i-2)
        push!(vs, 3.0 / 2g.Δx, -4.0 / 2g.Δx, 1.0 / 2g.Δx)
    end

end

function add_interior_point!(rs, cs, vs, i, x, y, g, p)

    push!(rs, i, i, i, i, i)
    push!(cs, i, i+1, i-1, i+g.nx, i-g.nx)

    push!(
        vs, 
        -(p(x+0.5g.Δx, y) + p(x-0.5g.Δx, y)) / g.Δx^2 - 
            (p(x, y+0.5g.Δy) + p(x, y-0.5g.Δy)) / g.Δy^2,
        p(x+0.5g.Δx, y) / g.Δx^2,
        p(x-0.5g.Δx, y) / g.Δx^2,
        p(x, y+0.5g.Δy) / g.Δy^2,
        p(x, y-0.5g.Δy) / g.Δy^2
    )

end

function generate_grid(xs, ys, p, bcs)

    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)

    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]

    nx = length(xs)
    ny = length(ys)
    nu = nx * ny

    g = Grid(
        xs, ys, 
        xmin, xmax, 
        ymin, ymax, 
        Δx, Δy, 
        nx, ny, nu
    )

    # Initialise the components of A
    rs = Int[]
    cs = Int[]
    vs = Float64[]
    
    b = zeros(g.nu)

    for i ∈ 1:g.nu 

        # Find the x and y coordinates of the current point
        x = g.xs[(i-1)%g.nx+1] 
        y = g.ys[Int(ceil(i/g.nx))]

        if in_corner(x, y, g)

            add_corner_point!(rs, cs, vs, i)

        elseif on_boundary(x, y, g)

            bc = get_boundary(x, y, g, bcs)
            add_boundary_point!(b, rs, cs, vs, i, x, y, g, bc)
        
        else
        
            add_interior_point!(rs, cs, vs, i, x, y, g, p)
        
        end

    end

    A = sparse(rs, cs, vs, g.nu, g.nu)

    return A, b

end