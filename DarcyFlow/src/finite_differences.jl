export Grid, BoundaryCondition, construct_grid, construct_A, construct_b

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

function construct_grid(
    xs::AbstractVector, 
    ys::AbstractVector
)::Grid

    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)

    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]

    nx = length(xs)
    ny = length(ys)
    nu = nx * ny

    return Grid(xs, ys, xmin, xmax, ymin, ymax, Δx, Δy, nx, ny, nu)

end

function get_coordinates(
    i::Int,
    g::Grid
)::Tuple{Real, Real}

    x = g.xs[(i-1)%g.nx+1] 
    y = g.ys[Int(ceil(i/g.nx))]
    return x, y

end

function in_corner(
    x::Real, 
    y::Real, 
    g::Grid
)::Bool

    return x ∈ [g.xmin, g.xmax] && y ∈ [g.ymin, g.ymax]

end

function on_boundary(
    x::Real, 
    y::Real, 
    g::Grid
)::Bool

    return x ∈ [g.xmin, g.xmax] || y ∈ [g.ymin, g.ymax]

end

function get_boundary(
    x::Real, 
    y::Real, 
    g::Grid, 
    bcs::Dict{Symbol, BoundaryCondition}
)::BoundaryCondition

    x == g.xmin && return bcs[:x0]
    x == g.xmax && return bcs[:x1]
    y == g.ymin && return bcs[:y0]
    y == g.ymax && return bcs[:y1]

    error("Point ($x, $y) is not on a boundary.")

end

function add_corner_point!(
    rs::Vector{Int}, 
    cs::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int
)::Nothing

    push!(rs, i)
    push!(cs, i)
    push!(vs, 1.0)

    return

end

function add_boundary_point!(
    rs::Vector{Int}, 
    cs::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int, 
    x::Real,
    y::Real,
    g::Grid, 
    bc::BoundaryCondition,
    ps::Interpolations.GriddedInterpolation
)::Nothing

    bc.type == :dirichlet && add_dirichlet_point!(rs, cs, vs, i)
    bc.type == :neumann && add_neumann_point!(rs, cs, vs, i, x, y, g, bc, ps)

    return

end

function add_dirichlet_point!(
    rs::Vector{Int}, 
    cs::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int
)::Nothing

    push!(rs, i)
    push!(cs, i)
    push!(vs, 1.0)

    return

end

function add_neumann_point!(
    rs::Vector{Int}, 
    cs::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int,
    x::Real, 
    y::Real, 
    g::Grid, 
    bc::BoundaryCondition,
    ps::Interpolations.GriddedInterpolation
)::Nothing

    push!(rs, i, i, i)

    bc.name == :x0 && push!(cs, i, i+1, i+2)
    bc.name == :x1 && push!(cs, i, i-1, i-2)
    bc.name == :y0 && push!(cs, i, i+g.nx, i+2g.nx)
    bc.name == :y1 && push!(cs, i, i-g.nx, i-2g.nx)å

    push!(vs, 3.0ps(x, y) / 2g.Δx, -4.0ps(x, y) / 2g.Δx, 1.0ps(x, y) / 2g.Δx)

    return

end

function add_interior_point!(
    rs::Vector{Int}, 
    cs::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int, 
    x::Real, 
    y::Real, 
    g::Grid, 
    ps::Interpolations.GriddedInterpolation
)::Nothing

    push!(rs, i, i, i, i, i)
    push!(cs, i, i+1, i-1, i+g.nx, i-g.nx)

    push!(
        vs,
        -(ps(x+0.5g.Δx, y) + ps(x-0.5g.Δx, y)) / g.Δx^2 - 
         (ps(x, y+0.5g.Δy) + ps(x, y-0.5g.Δy)) / g.Δy^2,
        ps(x+0.5g.Δx, y) / g.Δx^2,
        ps(x-0.5g.Δx, y) / g.Δx^2,
        ps(x, y+0.5g.Δy) / g.Δy^2,
        ps(x, y-0.5g.Δy) / g.Δy^2
    )

    return

end

function construct_A(
    g::Grid, 
    ps::AbstractMatrix, 
    bcs::Dict{Symbol, BoundaryCondition}
)::SparseMatrixCSC

    # Initialise the components of A
    rs = Int[]
    cs = Int[]
    vs = Vector{typeof(ps[1, 1])}(undef, 0)

    ps = interpolate((g.xs, g.ys), ps, Gridded(Linear()))

    for i ∈ 1:g.nu 

        x, y = get_coordinates(i, g)

        if in_corner(x, y, g)

            add_corner_point!(rs, cs, vs, i)

        elseif on_boundary(x, y, g)

            bc = get_boundary(x, y, g, bcs)
            add_boundary_point!(rs, cs, vs, i, x, y, g, bc, ps)
        
        else
        
            add_interior_point!(rs, cs, vs, i, x, y, g, ps)
        
        end

    end

    return sparse(rs, cs, vs, g.nu, g.nu)

end

function construct_b(
    g::Grid, 
    ps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition}
)::SparseVector

    # Initialise the components of b
    is = Int[]
    vs = Vector{typeof(ps[1, 1])}(undef, 0)

    for i ∈ 1:g.nu 

        x, y = get_coordinates(i, g)

        if on_boundary(x, y, g)
            bc = get_boundary(x, y, g, bcs)
            push!(is, i)
            push!(vs, bc.func(x, y))
        end

    end

    return sparsevec(is, vs, g.nu)

end