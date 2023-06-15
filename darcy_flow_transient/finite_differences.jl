using Interpolations
using LinearAlgebra
using SparseArrays

abstract type Grid end

struct SteadyStateGrid <: Grid

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

struct TimeVaryingGrid <: Grid
    
    xs::AbstractVector 
    ys::AbstractVector 
    ts::AbstractVector

    xmin::Real 
    xmax::Real 
    ymin::Real 
    ymax::Real
    tmax::Real
    
    Δx::Real 
    Δy::Real 
    Δt::Real 

    nx::Int 
    ny::Int 
    nt::Int
    nu::Int

end

struct BoundaryCondition
    name::Symbol
    type::Symbol
    func::Function
end

function construct_grid(
    xs::AbstractVector, 
    ys::AbstractVector,
    tmax::Union{<:Real, Nothing}=nothing,
    Δt::Union{<:Real, Nothing}=nothing
)::Grid

    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)

    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]

    nx = length(xs)
    ny = length(ys)
    nu = nx * ny

    if tmax === nothing 

        return SteadyStateGrid(
            xs, ys, 
            xmin, xmax, ymin, ymax, 
            Δx, Δy, 
            nx, ny, nu
        )

    end

    ts = 0.0:Δt:tmax 
    nt = length(ts)-1 # Don't include the initial time

    return TimeVaryingGrid(
        xs, ys, ts, 
        xmin, xmax, ymin, ymax, tmax, 
        Δx, Δy, Δt, 
        nx, ny, nt, nu
    )

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
    g::Grid, 
    bc::BoundaryCondition
)::Nothing

    bc.type == :dirichlet && add_dirichlet_point!(rs, cs, vs, i)
    bc.type == :neumann && add_neumann_point!(rs, cs, vs, i, g, bc)

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
    g::Grid, 
    bc::BoundaryCondition,
)::Nothing

    push!(rs, i, i, i)

    bc.name == :x0 && push!(cs, i, i+1, i+2)
    bc.name == :x1 && push!(cs, i, i-1, i-2)
    bc.name == :y0 && push!(cs, i, i+g.nx, i+2g.nx)
    bc.name == :y1 && push!(cs, i, i-g.nx, i-2g.nx)

    push!(vs, 3.0 / 2g.Δx, -4.0 / 2g.Δx, 1.0 / 2g.Δx)

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

    θ = 0.5

    push!(rs, i, i, i, i, i, i, i, i, i, i)

    # Add coefficients for points at previous time
    push!(cs, i-g.nu, i-g.nu+1, i-g.nu-1, i-g.nu+g.nx, i-g.nu-g.nx)
    push!(
        vs, 
        1 - (1-θ) * (g.Δt / g.Δx^2) * (ps(x+0.5g.Δx, y) + ps(x-0.5g.Δx, y)) +
          - (1-θ) * (g.Δt / g.Δy^2) * (ps(x, y+0.5g.Δy) + ps(x, y-0.5g.Δy)),
        (1-θ) * (g.Δt / g.Δx^2) * ps(x+0.5g.Δx, y),
        (1-θ) * (g.Δt / g.Δx^2) * ps(x-0.5g.Δx, y),
        (1-θ) * (g.Δt / g.Δy^2) * ps(x, y+0.5g.Δy),
        (1-θ) * (g.Δt / g.Δy^2) * ps(x, y-0.5g.Δy)
    )

    # Add coefficients for points at current time
    push!(cs, i, i+1, i-1, i+g.nx, i-g.nx)
    push!(
        vs, 
        -1 - θ * (g.Δt / g.Δx^2) * (ps(x+0.5g.Δx, y) + ps(x-0.5g.Δx, y)) +
           - θ * (g.Δt / g.Δy^2) * (ps(x, y+0.5g.Δy) + ps(x, y-0.5g.Δy)),
        θ * (g.Δt / g.Δx^2) * ps(x+0.5g.Δx, y),
        θ * (g.Δt / g.Δx^2) * ps(x-0.5g.Δx, y),
        θ * (g.Δt / g.Δy^2) * ps(x, y+0.5g.Δy),
        θ * (g.Δt / g.Δy^2) * ps(x, y-0.5g.Δy)
    )

    return

end

function construct_A(
    g::Grid, 
    ps::AbstractMatrix, 
    bcs::Dict{Symbol, BoundaryCondition}
)::SparseMatrixCSC

    ps = interpolate((g.xs, g.ys), ps, Gridded(Linear()))

    rs = Int[]
    cs = Int[]
    vs = Vector{typeof(ps[1, 1])}(undef, 0)

    # Form the matrix of previous points
    for i ∈ 1:g.nu
        
        push!(rs, i)
        push!(cs, i)
        push!(vs, 1.0)

    end

    # Form the matrix of current points
    for i ∈ 1:g.nu 

        x, y = get_coordinates(i, g)

        if in_corner(x, y, g)

            add_corner_point!(rs, cs, vs, g.nu+i)

        elseif on_boundary(x, y, g)

            bc = get_boundary(x, y, g, bcs)
            add_boundary_point!(rs, cs, vs, g.nu+i, g, bc)
        
        else
        
            add_interior_point!(rs, cs, vs, g.nu+i, x, y, g, ps)
        
        end

    end

    return sparse(rs, cs, vs, 2g.nu, 2g.nu)

end

function construct_b(
    g::Grid, 
    ps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition},
    u_p::AbstractMatrix
)::SparseVector

    is = Int[]
    vs = Vector{typeof(ps[1, 1])}(undef, 0)

    ps = interpolate((g.xs, g.ys), ps, Gridded(Linear()))

    push!(is, collect(1:g.nu)...)
    push!(vs, u_p...)

    for i ∈ 1:g.nu 

        x, y = get_coordinates(i, g)

        # Everything except for the boundaries is equal to 0
        if on_boundary(x, y, g)

            push!(is, i+g.nu)

            bc = get_boundary(x, y, g, bcs)
            if bc.type == :neumann
                push!(vs, bc.func(x, y) / ps(x, y))
            else 
                push!(vs, bc.func(x, y))
            end

        end

    end

    return sparsevec(is, vs, 2g.nu)

end