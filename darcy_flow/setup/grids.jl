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

    μ::Real

    function SteadyStateGrid(
        xs::AbstractVector,
        ys::AbstractVector,
        μ::Real=1.0,
    )::SteadyStateGrid 
        
        xmin, xmax = extrema(xs)
        ymin, ymax = extrema(ys)

        Δx = xs[2] - xs[1]
        Δy = ys[2] - ys[1]

        nx = length(xs)
        ny = length(ys)
        nu = nx * ny

        return new(
            xs, ys, 
            xmin, xmax, 
            ymin, ymax, 
            Δx, Δy, 
            nx, ny, nu,
            μ
        )

    end

end

struct TransientGrid <: Grid
    
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

    μ::Real 
    ϕ::Real
    c::Real

    function TransientGrid(
        xs::AbstractVector,
        ys::AbstractVector,
        tmax::Real,
        Δt::Real,
        μ::Real=1.0,
        ϕ::Real=1.0,
        c::Real=1.0
    )::TransientGrid 

        ts = 0.0:Δt:tmax 

        xmin, xmax = extrema(xs)
        ymin, ymax = extrema(ys)

        Δx = xs[2] - xs[1]
        Δy = ys[2] - ys[1]

        nx = length(xs)
        ny = length(ys)
        nt = length(ts)-1
        nu = nx * ny

        return new(
            xs, ys, ts, 
            xmin, xmax, 
            ymin, ymax, tmax, 
            Δx, Δy, Δt, 
            nx, ny, nt, nu,
            μ, ϕ, c
        )

    end

end

struct BoundaryCondition

    name::Symbol
    type::Symbol
    func::Function
    
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