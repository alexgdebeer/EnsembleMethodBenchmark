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

    function SteadyStateGrid(
        xs::AbstractVector,
        ys::AbstractVector
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
            nx, ny, nu
        )

    end

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

    function TimeVaryingGrid(
        xs::AbstractVector,
        ys::AbstractVector,
        tmax::Real,
        Δt::Real
    )::TimeVaryingGrid 

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
            nx, ny, nt, nu
        )

    end

end

struct BoundaryCondition

    name::Symbol
    type::Symbol
    func::Function
    
end