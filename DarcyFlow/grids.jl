using LinearAlgebra
using SparseArrays

"""Builds gradient operator."""
function build_∇h(nx::Real, Δx::Real)::SparseMatrixCSC

    # Inner points
    ∇hi_i = repeat(2:nx, inner=2)
    ∇hi_j = vcat([[i-1, i] for i ∈ 2:nx]...)
    ∇hi_v = repeat([-1, 1], outer=(nx-1))

    # Neumann boundaries (TODO: check)
    push!(∇hi_i, 1, 1, nx+1, nx+1)
    push!(∇hi_j, 1, 2, nx-1, nx)
    push!(∇hi_v, -1, 1, -1, 1)

    ∇hi = sparse(∇hi_i, ∇hi_j, ∇hi_v, nx+1, nx) / Δx
    Ii = sparse(I, nx, nx)

    ∇h = [kron(Ii, ∇hi); kron(∇hi, Ii)]
    return ∇h

end

"""Builds operator that interpolates between cells and faces."""
function build_A(nx::Real)::SparseMatrixCSC

    Ai_i = repeat(2:nx, inner=2)
    Ai_j = vcat([[i-1, i] for i ∈ 2:nx]...)
    Ai_v = fill(0.5, 2*(nx-1))

    push!(Ai_i, 1, nx+1)
    push!(Ai_j, 1, nx)
    push!(Ai_v, 1, 1)

    Ai = sparse(Ai_i, Ai_j, Ai_v, nx+1, nx)
    Ii = sparse(I, nx, nx)

    A = [kron(Ii, Ai); kron(Ai, Ii)]
    return A

end

"""Builds the operator that maps from model states to observations."""
function build_B(
    xs::AbstractVector,
    nx::Int,
    nt::Int,
    ny::Int,
    nyi::Int,
    x_obs::AbstractVector,
    y_obs::AbstractVector,
    t_obs_inds::AbstractVector 
)

    function get_cell_index(xi::Int, yi::Int)
        return xi + nx * (yi-1)
    end

    is = Int[]
    js = Int[]
    vs = Float64[]

    for (i, (x, y)) ∈ enumerate(zip(x_obs, y_obs))

        ix0 = findfirst(xs .> x) - 1
        iy0 = findfirst(xs .> y) - 1
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        x0, x1 = xs[ix0], xs[ix1]
        y0, y1 = xs[iy0], xs[iy1]

        inds = [(ix0, iy0), (ix0, iy1), (ix1, iy0), (ix1, iy1)]
        cell_inds = [get_cell_index(i...) for i ∈ inds]

        Z = (x1-x0) * (y1-y0)

        push!(is, i, i, i, i)
        push!(js, cell_inds...)
        push!(vs,
            (x1-x) * (y1-y) / Z, 
            (x-x0) * (y1-y) / Z, 
            (x1-x) * (y-y0) / Z, 
            (x-x0) * (y-y0) / Z
        )

    end

    Bi = sparse(is, js, vs, nyi, nx^2)
    # Bis = [i ∈ t_obs_inds ? Bi : spzeros(nyi, nx^2) for i ∈ 1:nt]

    B = spzeros(ny, nx^2 * nt)

    for (i, t) ∈ enumerate(t_obs_inds)
        ii = (i-1) * nyi
        jj = (t-1) * nx^2
        B[(ii+1):(ii+nyi), (jj+1):(jj+nx^2)] = Bi
    end

    return B, Bi

end

struct Grid 

    xs::AbstractVector
    ts::AbstractVector

    cxs::AbstractVector 
    cys::AbstractVector
    
    Δx::Real 
    Δt::Real

    nx::Int 
    nt::Int
    ny::Int
    nyi::Int

    t_obs_inds::AbstractVector

    ∇h::SparseMatrixCSC
    A::SparseMatrixCSC
    B::SparseMatrixCSC
    Bi::SparseMatrixCSC

    ϕ::Real 
    μ::Real
    c::Real
    u0::Real

    function Grid(
        xmax::Real,
        tmax::Real,
        Δx::Real,
        Δt::Real,
        x_obs::AbstractVector,
        y_obs::AbstractVector,
        t_obs::AbstractVector,
        ϕ::Real=1.0,
        μ::Real=1.0,
        c::Real=1.0,
        u0::Real=1.0
    )

        nx = Int(round(xmax / Δx))
        nt = Int(round(tmax / Δt))

        nyi = length(x_obs)
        ny = nyi * length(t_obs)

        xs = LinRange(0.5Δx, xmax-0.5Δx, nx)
        ts = LinRange(Δt, tmax, nt)

        cxs = repeat(xs, outer=nx)
        cys = repeat(xs, inner=nx)

        ∇h = build_∇h(nx, Δx)
        A = build_A(nx)

        t_obs_inds = [findfirst(ts .>= t) for t ∈ t_obs]
        B, Bi = build_B(xs, nx, nt, ny, nyi, x_obs, y_obs, t_obs_inds)

        return new(
            xs, ts, cxs, cys, Δx, Δt, 
            nx, nt, ny, nyi, t_obs_inds, 
            ∇h, A, B, Bi,
            ϕ, μ, c, u0
        )

    end

end