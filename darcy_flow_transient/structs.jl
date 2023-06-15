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