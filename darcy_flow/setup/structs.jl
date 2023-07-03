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

"""Calculates the value to scale a bump function by, such that the values of 
the function on a grid sum to 1."""
function normalising_constant(g::Grid, x::Real, y::Real, r::Real)::Real

    a = 0.0

    for gx ∈ g.xs, gy ∈ g.ys

        r_sq = (gx - x)^2 + (gy - y)^2
    
        if r_sq < r^2
            a += exp(-1/(r^2-r_sq))
        end
    
    end

    return a

end

struct DeltaWell 

    x::Real 
    y::Real
    t0::Real 
    t1::Real
    q::Real

end

struct BumpWell

    x::Real
    y::Real
    r::Real
    t0::Real
    t1::Real
    q::Real
    a::Real
    
    function BumpWell(
        g::Grid, 
        x::Real, 
        y::Real, 
        r::Real,
        t0::Real,
        t1::Real, 
        q::Real
    )

        a = normalising_constant(g, x, y, r)
        return new(x, y, r, t0, t1, q, a)
    
    end

end

function well_rate(w::DeltaWell, x::Real, y::Real, t::Real)::Real 

    if t < w.t0 || t > w.t1
        return 0.0
    end

    if abs(w.x - x) ≤ 1e-8 && abs(w.y - y) ≤ 1e-8
        return w.q 
    end

    return 0.0
    
end

function well_rate(w::BumpWell, x::Real, y::Real, t::Real)::Real

    if t < w.t0 || t > w.t1
        return 0.0
    end

    r_sq = (x - w.x)^2 + (y - w.y)^2

    if r_sq ≥ w.r^2
        return 0.0
    end

    return w.q * exp(-1/(w.r^2-r_sq)) / w.a

end