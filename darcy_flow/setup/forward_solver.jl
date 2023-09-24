using Interpolations
using LinearAlgebra
using LinearSolve
using SparseArrays

# TODO: ensure the boundary conditions are satisfied as part of the initial condition

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
    ps::Interpolations.GriddedInterpolation
)::Nothing

    push!(is, i, i, i, i, i)
    push!(js, i, i+1, i-1, i+g.nx, i-g.nx)

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
    is::Vector{Int}, 
    js::Vector{Int}, 
    vs::Vector{<:Real}, 
    i::Int, 
    x::Real, 
    y::Real, 
    g::TransientGrid, 
    ps::Interpolations.GriddedInterpolation
)::Nothing

    push!(is, i, i, i, i, i, i, i, i, i, i)

    # Add the coefficients for points at the previous time
    push!(js, i-g.nu, i-g.nu+1, i-g.nu-1, i-g.nu+g.nx, i-g.nu-g.nx)
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
    push!(js, i, i+1, i-1, i+g.nx, i-g.nx)
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

    is = Int[]
    js = Int[]
    vs = Vector{typeof(ps[1, 1])}(undef, 0)

    ps = interpolate((g.xs, g.ys), ps, Gridded(Linear()))

    for i ∈ 1:g.nu 

        x, y = get_coordinates(i, g)

        if in_corner(x, y, g)

            add_corner_point!(is, js, vs, i)

        elseif on_boundary(x, y, g)

            bc = get_boundary(x, y, g, bcs)
            add_boundary_point!(is, js, vs, i, g, bc)
        
        else
        
            add_interior_point!(is, js, vs, i, x, y, g, ps)
        
        end

    end

    return sparse(is, js, vs, g.nu, g.nu)

end

function construct_A(
    g::TransientGrid, 
    ps::AbstractMatrix, 
    bcs::Dict{Symbol, BoundaryCondition}
)::SparseMatrixCSC

    ps = interpolate((g.xs, g.ys), ps, Gridded(Linear()))

    is = Int[]
    js = Int[]
    vs = Vector{typeof(ps[1, 1])}(undef, 0)

    # Form the matrix associated with the previous points
    push!(is, collect(1:g.nu)...)
    push!(js, collect(1:g.nu)...)
    push!(vs, fill(1.0, g.nu)...)

    # Form the matrix of current points
    for i ∈ 1:g.nu 

        x, y = get_coordinates(i, g)

        if in_corner(x, y, g)

            add_corner_point!(is, js, vs, g.nu+i)

        elseif on_boundary(x, y, g)

            bc = get_boundary(x, y, g, bcs)
            add_boundary_point!(is, js, vs, g.nu+i, g, bc)
        
        else
        
            add_interior_point!(is, js, vs, g.nu+i, x, y, g, ps)
        
        end

    end

    return sparse(is, js, vs, 2g.nu, 2g.nu)

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

    u0 = reshape([bcs[:t0].func(x, y) for x ∈ g.xs for y ∈ g.ys], g.nx, g.ny)

    us = zeros(Real, g.nx, g.ny, g.nt+1)
    us[:,:,1] = u0

    A = construct_A(g, ps, bcs)

    for t ∈ 1:g.nt

        b = construct_b(g, ps, bcs, us[:,:,t], q, t)
        u = solve(LinearProblem(A, b))
        us[:,:,t+1] = reshape(u[g.nu+1:end], g.nx, g.ny)

    end

    return us

end