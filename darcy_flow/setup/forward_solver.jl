using Interpolations
using LinearAlgebra
using LinearSolve
using SparseArrays

const θ = 0.5 # Implicit solve parameter (Crank-Nicolson)

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
    g::SteadyStateGrid, 
    logps::Interpolations.GriddedInterpolation
)::Nothing

    x, y = g.ixs[i], g.iys[i]

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

    push!(P_is, i, i, i, i, i)
    push!(A_is, i, i, i, i, i)

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

    for i ∈ g.is_corner
        add_corner_point!(is, js, vs, i)
    end

    for (i, b) ∈ zip(g.is_bounds, g.bs_bounds)
        add_boundary_point!(is, js, vs, i, g, bcs[b])
    end

    for i ∈ g.is_inner
        add_interior_point!(is, js, vs, i, g, logps)
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
    Q::AbstractVector
)::SparseVector

    is = Int[]
    vs = Vector{typeof(logps[1, 1])}(undef, 0)

    logps = interpolate((g.xs, g.ys), logps, Gridded(Linear()))

    for (i, b) ∈ zip(g.is_bounds, g.bs_bounds)
        
        push!(is, i)

        if bcs[b].type == :neumann
            push!(vs, bcs[b].func(g.ixs[i], g.iys[i]) / 
                         10^logps(g.ixs[i], g.iys[i]))
        else 
            push!(vs, bcs[b].func(g.ixs[i], g.iys[i]))
        end

    end

    b = sparsevec(is, vs, g.nu) + Q
    return b

end

function construct_b(
    g::TransientGrid, 
    logps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition},
    Qs::AbstractMatrix,
    t::Int
)::SparseVector

    p = -1
    p_prev = -1
    initial = false

    if t ∈ g.well_change_inds
        p = findfirst(g.well_change_inds .== t)
        p_prev = p-1
        initial = p == 1
    elseif t-1 ∈ g.well_change_inds
        p = findfirst(g.well_change_inds .== t-1)
        p_prev = p
    end

    is = Int[]
    vs = Vector{typeof(logps[1, 1])}(undef, 0)

    logps = interpolate((g.xs, g.ys), logps, Gridded(Linear()))

    for (i, b) ∈ zip(g.is_bounds, g.bs_bounds)
        
        push!(is, i)

        if bcs[b].type == :neumann
            push!(vs, bcs[b].func(g.ixs[i], g.iys[i]) / 
                         10^logps(g.ixs[i], g.iys[i]))
        else 
            push!(vs, bcs[b].func(g.ixs[i], g.iys[i]))
        end

    end

    b = sparsevec(is, vs, g.nu)

    # Add forcing term
    if initial 
        b += θ * Qs[:, 1]
    else 
        b += (1-θ) * Qs[:, p_prev] + θ * Qs[:, p]
    end

    return b

end

function SciMLBase.solve(
    g::SteadyStateGrid,
    ps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition},
    Q::AbstractVector
)::AbstractVector

    A = construct_A(g, ps, bcs)
    b = construct_b(g, ps, bcs, Q)

    us = solve(LinearProblem(A, b))
    return us

end

function SciMLBase.solve(
    g::TransientGrid,
    logps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition},
    Qs::AbstractMatrix
)::AbstractVector

    u0 = [bcs[:t0].func(x, y) for x ∈ g.xs for y ∈ g.ys]
    us = zeros(typeof(logps[1, 1]), g.nx * g.ny, g.nt+1)
    us[:, 1] = u0

    P, A = construct_A(g, logps, bcs)
    b = construct_b(g, logps, bcs, Qs, 1)

    for t ∈ 1:g.nt

        us[:, t+1] = solve(LinearProblem(A, b-P*us[:, t]))

        if t ∈ g.well_change_inds || t+1 ∈ g.well_change_inds
            b = construct_b(g, logps, bcs, Qs, t+1)
        end

    end

    return vec(us)

end

function SciMLBase.solve(
    g::TransientGrid,
    logps::AbstractMatrix,
    bcs::Dict{Symbol, BoundaryCondition},
    Qs::AbstractMatrix,
    μ::AbstractVector,
    V_r::AbstractMatrix
)::AbstractVector

    u0 = [bcs[:t0].func(x, y) for x ∈ g.xs for y ∈ g.ys]
    us = zeros(typeof(logps[1, 1]), g.nx * g.ny, g.nt+1)
    us[:, 1] = u0

    P, A = construct_A(g, logps, bcs)
    A_r = V_r' * A * V_r

    b = construct_b(g, logps, bcs, Qs, 1)

    for t ∈ 1:g.nt

        b_r = V_r' * (b - P * us[:, t] - A * μ)
        us_r = solve(LinearProblem(A_r, b_r))
        us[:, t+1] = μ + V_r * us_r

        if t ∈ g.well_change_inds || t+1 ∈ g.well_change_inds
            b = construct_b(g, logps, bcs, Qs, t+1)
        end

    end

    return vec(us)

end