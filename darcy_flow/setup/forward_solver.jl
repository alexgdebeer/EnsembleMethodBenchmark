using Interpolations
using LinearAlgebra
using LinearSolve
using SparseArrays

# TODO: bring the steady-state solve into line with the transient solve

# Implicit solve parameter (Crank-Nicolson)
const θ = 0.5

function add_corner_point!(
    is::Vector{Int}, 
    js::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int
)::Nothing

    push!(is, i)
    push!(js, i)
    push!(vs, 1.0)

    return

end

function add_boundary_point!(
    is::Vector{Int}, 
    js::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int,
    g::Grid, 
    bc::BoundaryCondition
)::Nothing

    bc.type == :dirichlet && add_dirichlet_point!(is, js, vs, i)
    bc.type == :neumann && add_neumann_point!(is, js, vs, i, g, bc)

    return

end

function add_dirichlet_point!(
    is::Vector{Int}, 
    js::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int
)::Nothing

    push!(is, i)
    push!(js, i)
    push!(vs, 1.0)

    return

end

function add_neumann_point!(
    is::Vector{Int}, 
    js::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int,
    g::Grid, 
    bc::BoundaryCondition,
)::Nothing

    push!(is, i, i, i)

    if bc.name == :x0
        push!(js, i, i+1, i+2)
        push!(vs, 3.0 / 2g.Δx, -4.0 / 2g.Δx, 1.0 / 2g.Δx)
    elseif bc.name == :x1
        push!(js, i, i-1, i-2)
        push!(vs, -3.0 / 2g.Δx, 4.0 / 2g.Δx, -1.0 / 2g.Δx)
    elseif bc.name == :y0
        push!(js, i, i+g.nx, i+2g.nx)
        push!(vs, 3.0 / 2g.Δy, -4.0 / 2g.Δy, 1.0 / 2g.Δy)
    elseif bc.name == :y1 
        push!(js, i, i-g.nx, i-2g.nx)
        push!(vs, -3.0 / 2g.Δy, 4.0 / 2g.Δy, -1.0 / 2g.Δy)
    end

    return

end

function add_interior_point!(
    is::Vector{Int}, 
    js::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int, 
    x::Real, 
    y::Real, 
    g::SteadyStateGrid, 
    logps::Interpolations.GriddedInterpolation
)::Nothing

    push!(is, i, i, i, i, i)
    push!(js, i, i+1, i-1, i+g.nx, i-g.nx)

    push!(
        vs,
        (1.0 / (g.μ * g.Δx^2)) * (10^logps(x+0.5g.Δx, y) + 10^logps(x-0.5g.Δx, y)) + 
        (1.0 / (g.μ * g.Δy^2)) * (10^logps(x, y+0.5g.Δy) + 10^logps(x, y-0.5g.Δy)),
        -(1.0 / (g.μ * g.Δx^2)) * 10^logps(x+0.5g.Δx, y),
        -(1.0 / (g.μ * g.Δx^2)) * 10^logps(x-0.5g.Δx, y),
        -(1.0 / (g.μ * g.Δy^2)) * 10^logps(x, y+0.5g.Δy),
        -(1.0 / (g.μ * g.Δy^2)) * 10^logps(x, y-0.5g.Δy)
    )

    return

end

function add_interior_point!(
    is::Vector{Int}, 
    js::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int,
    g::TransientGrid, 
    logps::Interpolations.GriddedInterpolation
)::Nothing

    x, y = g.ixs[i%g.nu], g.iys[i%g.nu]

    push!(is, fill(i, 10)...)

    # Add the coefficients for points at the previous time
    push!(js, i-g.nu, i-g.nu+1, i-g.nu-1, i-g.nu+g.nx, i-g.nu-g.nx)
    push!(
        vs, 
        -(g.ϕ*g.c / g.Δt) +
          ((1-θ) / (g.μ * g.Δx^2)) * (10^logps(x+0.5g.Δx, y) + 10^logps(x-0.5g.Δx, y)) +
          ((1-θ) / (g.μ * g.Δy^2)) * (10^logps(x, y+0.5g.Δy) + 10^logps(x, y-0.5g.Δy)),
        -((1-θ) / (g.μ * g.Δx^2)) * 10^logps(x+0.5g.Δx, y),
        -((1-θ) / (g.μ * g.Δx^2)) * 10^logps(x-0.5g.Δx, y),
        -((1-θ) / (g.μ * g.Δy^2)) * 10^logps(x, y+0.5g.Δy),
        -((1-θ) / (g.μ * g.Δy^2)) * 10^logps(x, y-0.5g.Δy)
    )

    # Add the coefficients for points at the current time
    push!(js, i, i+1, i-1, i+g.nx, i-g.nx)
    push!(
        vs, 
        (g.ϕ*g.c / g.Δt) +
          (θ / (g.μ * g.Δx^2)) * (10^logps(x+0.5g.Δx, y) + 10^logps(x-0.5g.Δx, y)) +
          (θ / (g.μ * g.Δy^2)) * (10^logps(x, y+0.5g.Δy) + 10^logps(x, y-0.5g.Δy)),
        -(θ / (g.μ * g.Δx^2)) * 10^logps(x+0.5g.Δx, y),
        -(θ / (g.μ * g.Δx^2)) * 10^logps(x-0.5g.Δx, y),
        -(θ / (g.μ * g.Δy^2)) * 10^logps(x, y+0.5g.Δy),
        -(θ / (g.μ * g.Δy^2)) * 10^logps(x, y-0.5g.Δy)
    )

    return

end

function construct_A(
    g::SteadyStateGrid, 
    logps::AbstractMatrix, 
    bcs::Dict{Symbol, BoundaryCondition}
)::SparseMatrixCSC

    is = Int[]
    js = Int[]
    vs = Vector{typeof(logps[1, 1])}(undef, 0)

    logps = interpolate((g.xs, g.ys), logps, Gridded(Linear()))

    for i ∈ 1:g.nu 

        x, y = get_coordinates(i, g)

        if in_corner(x, y, g)

            add_corner_point!(is, js, vs, i)

        elseif on_boundary(x, y, g)

            bc = get_boundary(x, y, g, bcs)
            add_boundary_point!(is, js, vs, i, g, bc)
        
        else
        
            add_interior_point!(is, js, vs, i, x, y, g, logps)
        
        end

    end

    return sparse(is, js, vs, g.nu, g.nu)

end

function construct_A(
    g::TransientGrid, 
    logps::AbstractMatrix, 
    bcs::Dict{Symbol, BoundaryCondition}
)::SparseMatrixCSC

    logps = interpolate((g.xs, g.ys), logps, Gridded(Linear()))

    is = Int[]
    js = Int[]
    vs = Vector{typeof(logps[1, 1])}(undef, 0)

    # Form the matrix associated with the previous points
    push!(is, collect(1:g.nu)...)
    push!(js, collect(1:g.nu)...)
    push!(vs, fill(1.0, g.nu)...)

    # Form the matrix of current points
    for i ∈ g.is_corner 
        add_corner_point!(is, js, vs, g.nu+i)
    end

    for (i, b) ∈ zip(g.is_bounds, g.bs_bounds)
        add_boundary_point!(is, js, vs, g.nu+i, g, bcs[b])
    end

    for i ∈ g.is_inner
        add_interior_point!(is, js, vs, g.nu+i, g, logps)
    end

    return sparse(is, js, vs, 2g.nu, 2g.nu)

end

function construct_b(
    g::SteadyStateGrid, 
    logps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition},
    q::Function
)::SparseVector

    is = Int[]
    vs = Vector{typeof(logps[1, 1])}(undef, 0)

    logps = interpolate((g.xs, g.ys), logps, Gridded(Linear()))

    for i ∈ 1:g.nu 

        x, y = get_coordinates(i, g)

        if on_boundary(x, y, g)

            push!(is, i)

            bc = get_boundary(x, y, g, bcs)
            if bc.type == :neumann
                push!(vs, bc.func(x, y) / logps(x, y))
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
    logps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition},
    u_p::AbstractMatrix,
    q::Function,
    t::Real
)::SparseVector

    is = Int[]
    vs = Vector{typeof(logps[1, 1])}(undef, 0)

    logps = interpolate((g.xs, g.ys), logps, Gridded(Linear()))

    push!(is, collect(1:g.nu)...)
    push!(vs, u_p...)

    for (i, b) ∈ zip(g.is_bounds, g.bs_bounds)
        
        push!(is, i+g.nu)

        if bcs[b].type == :neumann
            push!(vs, bcs[b].func(g.ixs[i], g.iys[i]) / 10^logps(g.ixs[i], g.iys[i]))
        else 
            push!(vs, bcs[b].func(g.ixs[i], g.iys[i]))
        end

    end

    # Add source term
    for i ∈ g.is_inner

        push!(is, i+g.nu)
        if t == 1
            push!(vs, θ * q(g.ixs[i], g.iys[i], g.ts[t+1]))
        else
            push!(vs, (1-θ) * q(g.ixs[i], g.iys[i], g.ts[t]) + θ * q(g.ixs[i], g.iys[i], g.ts[t+1]))
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
    logps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition},
    q::Function
)::AbstractArray

    u0 = reshape([bcs[:t0].func(x, y) for x ∈ g.xs for y ∈ g.ys], g.nx, g.ny)

    us = zeros(Real, g.nx, g.ny, g.nt+1)
    us[:,:,1] = u0

    A = construct_A(g, logps, bcs)

    for t ∈ 1:g.nt

        b = construct_b(g, logps, bcs, us[:,:,t], q, t)
        u = solve(LinearProblem(A, b))
        us[:,:,t+1] = reshape(u[g.nu+1:end], g.nx, g.ny)

    end

    return us

end