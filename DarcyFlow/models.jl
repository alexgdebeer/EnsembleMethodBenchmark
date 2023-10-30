using SparseArrays
# TODO: change y to d?

"""Well that has been mollified using a bump function of radius r."""
struct Well

    cx::Real
    cy::Real
    r::Real

    rates::Tuple
    Z::Real
    
    function Well(
        g::Grid, 
        cx::Real, 
        cy::Real, 
        r::Real,
        rates::Tuple
    )::Well

        """Calculates the value to scale the bump function by, such that 
        the values of the function on the model grid sum to 1."""
        function normalising_constant(
            g::Grid, 
            cx::Real, 
            cy::Real, 
            r::Real
        )::Real
        
            Z = 0.0
            for (x, y) ∈ zip(g.cxs, g.cys)
                if (r_sq = (x-cx)^2 + (y-cy)^2) < r^2
                    Z += exp(-1/(r^2-r_sq))
                end
            end
        
            return Z
        
        end

        Z = normalising_constant(g, cx, cy, r)

        return new(cx, cy, r, rates, Z)
    
    end

end

abstract type AbstractModel end

struct Model <: AbstractModel

    ϕ::Real
    μ::Real
    c::Real
    u0::Real

    Q::SparseMatrixCSC
    B::SparseMatrixCSC
    B_wells::SparseMatrixCSC

    ny::Int
    nyi::Int

    function Model(
        g::Grid,
        ϕ::Real, 
        μ::Real, 
        c::Real, 
        u0::Real,
        wells::AbstractVector{Well},
        well_change_times::AbstractVector,
        x_obs::AbstractVector,
        y_obs::AbstractVector,
        t_obs::AbstractVector
    )

        nyi = length(x_obs)
        ny = nyi * length(t_obs)

        t_obs_inds = [findfirst(g.ts .>= t) for t ∈ t_obs]

        Q = build_Q(g, wells, well_change_times)
        B, B_wells = build_B(g, ny, nyi, x_obs, y_obs, t_obs_inds)

        return new(ϕ, μ, c, u0, Q, B, B_wells, ny, nyi)

    end

end

struct ReducedOrderModel <: AbstractModel

    ϕ::Real
    μ::Real
    c::Real
    u0::Real

    Q::SparseMatrixCSC
    B::SparseMatrixCSC
    B_wells::SparseMatrixCSC
    BV_r::SparseMatrixCSC

    μ_ui::AbstractVector
    V_ri::AbstractMatrix
    
    μ_e::AbstractVector
    Γ_e::AbstractMatrix 
    Γ_e_inv::AbstractMatrix
    L_e::AbstractMatrix

    nu_r::Int
    ny::Int
    nyi::Int

    function ReducedOrderModel(
        g::Grid,
        ϕ::Real, 
        μ::Real, 
        c::Real, 
        u0::Real,
        wells::AbstractVector{Well},
        well_change_times::AbstractVector,
        x_obs::AbstractVector,
        y_obs::AbstractVector,
        t_obs::AbstractVector,
        μ_ui::AbstractVector,
        V_ri::AbstractMatrix,
        μ_e::AbstractVector,
        Γ_e::AbstractMatrix
    )

        nu_r = size(V_ri, 2)
        nyi = length(x_obs)
        ny = nyi * length(t_obs)

        t_obs_inds = [findfirst(g.ts .>= t) for t ∈ t_obs]

        Q = build_Q(g, wells, well_change_times)
        B, B_wells = build_B(g, ny, nyi, x_obs, y_obs, t_obs_inds)

        V_r = sparse(kron(sparse(I, g.nt, g.nt), V_ri))
        BV_r = B * V_r

        Γ_e_inv = Hermitian(inv(Γ_e))
        L_e = cholesky(Γ_e_inv).U

        return new(
            ϕ, μ, c, u0, 
            Q, B, B_wells, BV_r, 
            μ_ui, V_ri, 
            μ_e, Γ_e, Γ_e_inv, L_e,
            nu_r, ny, nyi
        )

    end

end

"""Builds the operator that maps from model states to observations."""
function build_B(
    g::Grid,
    ny::Int,
    nyi::Int,
    x_obs::AbstractVector,
    y_obs::AbstractVector,
    t_obs_inds::AbstractVector 
)::Tuple{SparseMatrixCSC, SparseMatrixCSC}

    function get_cell_index(xi::Int, yi::Int)
        return xi + g.nx * (yi-1)
    end

    is = Int[]
    js = Int[]
    vs = Float64[]

    for (i, (x, y)) ∈ enumerate(zip(x_obs, y_obs))

        ix0 = findfirst(g.xs .> x) - 1
        iy0 = findfirst(g.xs .> y) - 1
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        x0, x1 = g.xs[ix0], g.xs[ix1]
        y0, y1 = g.xs[iy0], g.xs[iy1]

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

    B = spzeros(ny, g.nx^2 * g.nt)
    Bi = sparse(is, js, vs, nyi, g.nx^2)

    for (i, t) ∈ enumerate(t_obs_inds)
        ii = (i-1) * nyi
        jj = (t-1) * g.nx^2
        B[(ii+1):(ii+nyi), (jj+1):(jj+g.nx^2)] = Bi
    end

    B_wells = blockdiag([Bi for _ ∈ 1:grid_c.nt]...)

    return B, B_wells

end

"""Builds the forcing term at each time index."""
function build_Q(
    g::Grid,
    wells::AbstractVector{Well},
    well_change_times::AbstractVector 
)::SparseMatrixCSC

    Q_i = Int[]
    Q_j = Int[]
    Q_v = Float64[]

    time_inds = [findlast(well_change_times .<= t + 1e-8) for t ∈ g.ts]

    for (i, (x, y)) ∈ enumerate(zip(g.cxs, g.cys))
        for w ∈ wells 
            if (dist_sq = (x-w.cx)^2 + (y-w.cy)^2) < w.r^2
                for (j, q) ∈ enumerate(w.rates[time_inds])
                    push!(Q_i, i)
                    push!(Q_j, j)
                    push!(Q_v, q * exp(-1/(q^2-dist_sq)) / w.Z)
                end
            end
        end
    end

    Q = sparse(Q_i, Q_j, Q_v, g.nx^2, g.nt)
    return Q

end