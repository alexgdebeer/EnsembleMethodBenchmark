using Interpolations
using LinearAlgebra
using LinearSolve
using SciMLBase
using SparseArrays

# Implicit solve parameter (Crank-Nicolson)
const θ = 0.5

# TODO: ensure the boundary conditions are satisfied as part of the initial condition

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

    if bc.name == :x0
        push!(cs, i, i+1, i+2)
        push!(vs, 3.0 / 2g.Δx, -4.0 / 2g.Δx, 1.0 / 2g.Δx)
    elseif bc.name == :x1
        push!(cs, i, i-1, i-2)
        push!(vs, -3.0 / 2g.Δx, 4.0 / 2g.Δx, -1.0 / 2g.Δx)
    elseif bc.name == :y0
        push!(cs, i, i+g.nx, i+2g.nx)
        push!(vs, 3.0 / 2g.Δy, -4.0 / 2g.Δy, 1.0 / 2g.Δy)
    elseif bc.name == :y1 
        push!(cs, i, i-g.nx, i-2g.nx)
        push!(vs, -3.0 / 2g.Δy, 4.0 / 2g.Δy, -1.0 / 2g.Δy)
    end

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
        (1.0 / (g.μ * g.Δx^2)) * (ps(x+0.5g.Δx, y) + ps(x-0.5g.Δx, y)) + 
        (1.0 / (g.μ * g.Δy^2)) * (ps(x, y+0.5g.Δy) + ps(x, y-0.5g.Δy)),
        -(1.0 / (g.μ * g.Δx^2)) * ps(x+0.5g.Δx, y),
        -(1.0 / (g.μ * g.Δx^2)) * ps(x-0.5g.Δx, y),
        -(1.0 / (g.μ * g.Δy^2)) * ps(x, y+0.5g.Δy),
        -(1.0 / (g.μ * g.Δy^2)) * ps(x, y-0.5g.Δy)
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
    g::TransientGrid, 
    ps::Interpolations.GriddedInterpolation
)::Nothing

    push!(rs, i, i, i, i, i, i, i, i, i, i)

    # Add the coefficients for points at the previous time
    push!(cs, i-g.nu, i-g.nu+1, i-g.nu-1, i-g.nu+g.nx, i-g.nu-g.nx)
    push!(
        vs, 
        -(g.ϕ*g.c / g.Δt) +
          ((1-θ) / (g.μ * g.Δx^2)) * (ps(x+0.5g.Δx, y) + ps(x-0.5g.Δx, y)) +
          ((1-θ) / (g.μ * g.Δy^2)) * (ps(x, y+0.5g.Δy) + ps(x, y-0.5g.Δy)),
        -((1-θ) / (g.μ * g.Δx^2)) * ps(x+0.5g.Δx, y),
        -((1-θ) / (g.μ * g.Δx^2)) * ps(x-0.5g.Δx, y),
        -((1-θ) / (g.μ * g.Δy^2)) * ps(x, y+0.5g.Δy),
        -((1-θ) / (g.μ * g.Δy^2)) * ps(x, y-0.5g.Δy)
    )

    # Add the coefficients for points at the current time
    push!(cs, i, i+1, i-1, i+g.nx, i-g.nx)
    push!(
        vs, 
        (g.ϕ*g.c / g.Δt) +
          (θ / (g.μ * g.Δx^2)) * (ps(x+0.5g.Δx, y) + ps(x-0.5g.Δx, y)) +
          (θ / (g.μ * g.Δy^2)) * (ps(x, y+0.5g.Δy) + ps(x, y-0.5g.Δy)),
        -(θ / (g.μ * g.Δx^2)) * ps(x+0.5g.Δx, y),
        -(θ / (g.μ * g.Δx^2)) * ps(x-0.5g.Δx, y),
        -(θ / (g.μ * g.Δy^2)) * ps(x, y+0.5g.Δy),
        -(θ / (g.μ * g.Δy^2)) * ps(x, y-0.5g.Δy)
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
    g::TransientGrid, 
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
    bcs::Dict{Symbol, BoundaryCondition},
    q::Function
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

        else 

            push!(is, i)
            push!(vs, q(x, y))

        end

    end

    return sparsevec(is, vs, g.nu)

end

function construct_b(
    g::TransientGrid, 
    ps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition},
    u_p::AbstractMatrix,
    q::Function,
    t::Real
)::SparseVector

    is = Int[]
    vs = Vector{typeof(ps[1, 1])}(undef, 0)

    ps = interpolate((g.xs, g.ys), ps, Gridded(Linear()))

    push!(is, collect(1:g.nu)...)
    push!(vs, u_p...)

    for i ∈ 1:g.nu 

        x, y = get_coordinates(i, g)

        if on_boundary(x, y, g) && !in_corner(x, y, g)

            push!(is, i+g.nu)

            bc = get_boundary(x, y, g, bcs)
            if bc.type == :neumann
                push!(vs, bc.func(x, y) / ps(x, y))
            else 
                push!(vs, bc.func(x, y))
            end

        else

            # Add source term
            push!(is, i+g.nu)
            if t == 1
                push!(vs, θ * q(x, y, g.ts[t+1]))
            else
                push!(vs, (1-θ) * q(x, y, g.ts[t]) + θ * q(x, y, g.ts[t+1]))
            end
        
        end

    end

    return sparsevec(is, vs, 2g.nu)

end

function SciMLBase.solve(
    g::SteadyStateGrid,
    ps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition},
    q::Function
)::AbstractMatrix

    A = construct_A(g, ps, bcs)
    b = construct_b(g, ps, bcs, q)

    us = solve(LinearProblem(A, b))
    us = reshape(us, g.nx, g.ny)

    return us

end

function SciMLBase.solve(
    g::TransientGrid,
    ps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition},
    q::Function
)::AbstractArray

    u0 = [bcs[:t0].func(x, y) for x ∈ g.xs for y ∈ g.ys]

    us = zeros(g.nx, g.ny, g.nt+1)
    us[:,:,1] = u0

    A = construct_A(g, ps, bcs)

    for t ∈ 1:g.nt

        b = construct_b(g, ps, bcs, us[:,:,t], q, t)
        u = solve(LinearProblem(A, b))
        us[:,:,t+1] = reshape(u[g.nu+1:end], g.nx, g.ny)

    end

    return us

end