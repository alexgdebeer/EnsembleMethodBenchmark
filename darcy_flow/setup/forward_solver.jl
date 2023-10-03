using Interpolations
using LinearAlgebra
using LinearSolve
using SparseArrays

# TODO: bring the steady-state solve into line with the transient solve
# TODO: modify the transient solve to only modify b at the times it needs to be

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
    A_is::Vector{Int}, 
    A_js::Vector{Int}, 
    A_vs::Vector{<:Real},
    P_is::Vector{Int},
    P_js::Vector{Int},
    P_vs::Vector{<:Real}, 
    i::Int,
    g::TransientGrid, 
    logps::Interpolations.GriddedInterpolation
)::Nothing

    x, y = g.ixs[i], g.iys[i]

    push!(P_is, fill(i, 5)...)
    push!(A_is, fill(i, 5)...)

    push!(P_js, i, i+1, i-1, i+g.nx, i-g.nx)
    push!(A_js, i, i+1, i-1, i+g.nx, i-g.nx)

    # Previous timestep
    push!(
        P_vs, 
        -(g.ϕ*g.c / g.Δt) +
          ((1-θ) / (g.μ * g.Δx^2)) * (10^logps(x+0.5g.Δx, y) + 10^logps(x-0.5g.Δx, y)) +
          ((1-θ) / (g.μ * g.Δy^2)) * (10^logps(x, y+0.5g.Δy) + 10^logps(x, y-0.5g.Δy)),
        -((1-θ) / (g.μ * g.Δx^2)) * 10^logps(x+0.5g.Δx, y),
        -((1-θ) / (g.μ * g.Δx^2)) * 10^logps(x-0.5g.Δx, y),
        -((1-θ) / (g.μ * g.Δy^2)) * 10^logps(x, y+0.5g.Δy),
        -((1-θ) / (g.μ * g.Δy^2)) * 10^logps(x, y-0.5g.Δy)
    )

    # Current timestep
    push!(
        A_vs, 
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
)::Tuple{SparseMatrixCSC, SparseMatrixCSC}

    logps = interpolate((g.xs, g.ys), logps, Gridded(Linear()))

    # Previous timestep
    P_is = Int[]
    P_js = Int[]
    P_vs = Vector{typeof(logps[1, 1])}(undef, 0)

    # Current timestep
    A_is = Int[]
    A_js = Int[]
    A_vs = Vector{typeof(logps[1, 1])}(undef, 0)

    for i ∈ g.is_corner 
        add_corner_point!(A_is, A_js, A_vs, i)
    end

    for (i, b) ∈ zip(g.is_bounds, g.bs_bounds)
        add_boundary_point!(A_is, A_js, A_vs, i, g, bcs[b])
    end

    for i ∈ g.is_inner
        add_interior_point!(A_is, A_js, A_vs, P_is, P_js, P_vs, i, g, logps)
    end

    P = sparse(P_is, P_js, P_vs, g.nu, g.nu)
    A = sparse(A_is, A_js, A_vs, g.nu, g.nu)

    return P, A

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
    P::AbstractMatrix,
    q::Function,
    t::Real
)::SparseVector

    is = Int[]
    vs = Vector{typeof(logps[1, 1])}(undef, 0)

    logps = interpolate((g.xs, g.ys), logps, Gridded(Linear()))

    for (i, b) ∈ zip(g.is_bounds, g.bs_bounds)
        
        push!(is, i)

        if bcs[b].type == :neumann
            push!(vs, bcs[b].func(g.ixs[i], g.iys[i]) / 10^logps(g.ixs[i], g.iys[i]))
        else 
            push!(vs, bcs[b].func(g.ixs[i], g.iys[i]))
        end

    end

    # Add source term
    for i ∈ g.is_inner

        push!(is, i)
        if t == 1
            push!(vs, θ * q(g.ixs[i], g.iys[i], g.ts[t+1]))
        else
            push!(vs, (1-θ) * q(g.ixs[i], g.iys[i], g.ts[t]) + θ * q(g.ixs[i], g.iys[i], g.ts[t+1]))
        end

    end

    return sparsevec(is, vs, g.nu) - P*vec(u_p)

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

    us = zeros(typeof(logps[1, 1]), g.nx, g.ny, g.nt+1)
    us[:,:,1] = u0

    P, A = construct_A(g, logps, bcs)

    for t ∈ 1:g.nt

        b = construct_b(g, logps, bcs, us[:,:,t], P, q, t)
        u = solve(LinearProblem(A, b))
        us[:,:,t+1] = reshape(u, g.nx, g.ny)

    end

    return us

end