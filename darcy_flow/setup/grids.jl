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

    is_corner::AbstractVector 
    is_bounds::AbstractVector 
    bs_bounds::AbstractVector
    is_inner::AbstractVector

    nx::Int
    ny::Int
    nu::Int

    μ::Real

    function SteadyStateGrid(
        xs::AbstractVector,
        ys::AbstractVector,
        μ::Real=1.0,
    )::SteadyStateGrid 

        function add_point_type!(i, x, y)

            if x ∈ [xmin, xmax] && y ∈ [ymin, ymax]
                push!(is_corner, i)
                return
            end

            if x ∈ [xmin, xmax] || y ∈ [ymin, ymax]
                x == xmin && push!(bs_bounds, :x0)
                x == xmax && push!(bs_bounds, :x1)
                y == ymin && push!(bs_bounds, :y0)
                y == ymax && push!(bs_bounds, :y1)
                push!(is_bounds, i)
                return
            end
            
            push!(is_inner, i)
            return

        end
        
        xmin, xmax = extrema(xs)
        ymin, ymax = extrema(ys)

        Δx = xs[2] - xs[1]
        Δy = ys[2] - ys[1]

        nx = length(xs)
        ny = length(ys)
        nu = nx * ny

        ixs = repeat(xs, outer=ny)
        iys = repeat(ys, inner=nx)

        is_corner = []
        is_bounds = []
        bs_bounds = []
        is_inner  = []

        for (i, (x, y)) ∈ enumerate(zip(ixs, iys))
            add_point_type!(i, x, y)
        end

        return new(
            xs, ys, 
            xmin, xmax, 
            ymin, ymax, 
            Δx, Δy, 
            is_corner, is_bounds, bs_bounds, is_inner,
            nx, ny, nu,
            μ
        )

    end

end

struct TransientGrid <: Grid
    
    xs::AbstractVector 
    ys::AbstractVector 
    ts::AbstractVector

    ixs::AbstractVector 
    iys::AbstractVector

    xmin::Real 
    xmax::Real 
    ymin::Real 
    ymax::Real
    tmax::Real
    
    Δx::Real 
    Δy::Real 
    Δt::Real 

    is_corner::AbstractVector 
    is_bounds::AbstractVector 
    bs_bounds::AbstractVector
    is_inner::AbstractVector

    nx::Int 
    ny::Int 
    nt::Int
    nu::Int

    well_periods::Tuple

    μ::Real 
    ϕ::Real
    c::Real

    function TransientGrid(
        xs::AbstractVector,
        ys::AbstractVector,
        tmax::Real,
        Δt::Real,
        well_periods::Tuple,
        μ::Real=1.0,
        ϕ::Real=1.0,
        c::Real=1.0,
    )::TransientGrid 

        function add_point_type!(i, x, y)

            if x ∈ [xmin, xmax] && y ∈ [ymin, ymax]
                push!(is_corner, i)
                return
            end

            if x ∈ [xmin, xmax] || y ∈ [ymin, ymax]
                x == xmin && push!(bs_bounds, :x0)
                x == xmax && push!(bs_bounds, :x1)
                y == ymin && push!(bs_bounds, :y0)
                y == ymax && push!(bs_bounds, :y1)
                push!(is_bounds, i)
                return
            end
            
            push!(is_inner, i)
            return

        end

        ts = 0.0:Δt:tmax 

        xmin, xmax = extrema(xs)
        ymin, ymax = extrema(ys)

        Δx = xs[2] - xs[1]
        Δy = ys[2] - ys[1]

        nx = length(xs)
        ny = length(ys)
        nt = length(ts)-1
        nu = nx * ny

        ixs = repeat(xs, outer=ny)
        iys = repeat(ys, inner=nx)

        is_corner = []
        is_bounds = []
        bs_bounds = []
        is_inner  = []

        for (i, (x, y)) ∈ enumerate(zip(ixs, iys))
            add_point_type!(i, x, y)
        end

        return new(
            xs, ys, ts, 
            ixs, iys,
            xmin, xmax, 
            ymin, ymax, tmax, 
            Δx, Δy, Δt, 
            is_corner, is_bounds, bs_bounds, is_inner,
            nx, ny, nt, nu,
            well_periods,
            μ, ϕ, c
        )

    end

end

struct BoundaryCondition

    name::Symbol
    type::Symbol
    func::Function
    
end