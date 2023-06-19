using Interpolations
using LinearAlgebra
using LinearSolve
using SciMLBase
using SparseArrays

# TODO: make θ an input somewhere...

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
    g::SteadyStateGrid, 
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

function add_interior_point!(
    rs::Vector{Int}, 
    cs::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int, 
    x::Real, 
    y::Real, 
    g::TimeVaryingGrid, 
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
    g::SteadyStateGrid, 
    ps::AbstractMatrix, 
    bcs::Dict{Symbol, BoundaryCondition}
)::SparseMatrixCSC

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
            add_boundary_point!(rs, cs, vs, i, g, bc)
        
        else
        
            add_interior_point!(rs, cs, vs, i, x, y, g, ps)
        
        end

    end

    return sparse(rs, cs, vs, g.nu, g.nu)

end

function construct_A(
    g::TimeVaryingGrid, 
    ps::AbstractMatrix, 
    bcs::Dict{Symbol, BoundaryCondition}
)::SparseMatrixCSC

    ps = interpolate((g.xs, g.ys), ps, Gridded(Linear()))

    rs = Int[]
    cs = Int[]
    vs = Vector{typeof(ps[1, 1])}(undef, 0)

    # Form the matrix associated with the previous points
    push!(rs, collect(1:g.nu)...)
    push!(cs, collect(1:g.nu)...)
    push!(vs, fill(1.0, g.nu)...)

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
    g::SteadyStateGrid, 
    ps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition}
)::SparseVector

    is = Int[]
    vs = Vector{typeof(ps[1, 1])}(undef, 0)

    ps = interpolate((g.xs, g.ys), ps, Gridded(Linear()))

    for i ∈ 1:g.nu 

        x, y = get_coordinates(i, g)

        if on_boundary(x, y, g)

            push!(is, i)

            bc = get_boundary(x, y, g, bcs)
            if bc.type == :neumann
                push!(vs, bc.func(x, y) / ps(x, y))
            else 
                push!(vs, bc.func(x, y))
            end

        end

    end

    return sparsevec(is, vs, g.nu)

end

function construct_b(
    g::TimeVaryingGrid, 
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

function SciMLBase.solve(
    g::SteadyStateGrid,
    ps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition}
)::AbstractMatrix

    A = construct_A(g, ps, bcs)
    b = construct_b(g, ps, bcs)

    us = solve(LinearProblem(A, b))
    us = reshape(us, g.nx, g.ny)

    return us

end

function SciMLBase.solve(
    g::TimeVaryingGrid,
    ps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition}
)::AbstractArray

    u0 = [bcs[:t0].func(x, y) for x ∈ g.xs for y ∈ g.ys]

    us = zeros(g.nx, g.ny, g.nt+1)
    us[:,:,1] = u0

    A = construct_A(g, ps, bcs)

    for t ∈ 1:g.nt

        b = construct_b(g, ps, bcs, us[:,:,t])
        u = solve(LinearProblem(A, b))
        us[:,:,t+1] = reshape(u[g.nu+1:end], g.nx, g.ny)

    end

    return us

end