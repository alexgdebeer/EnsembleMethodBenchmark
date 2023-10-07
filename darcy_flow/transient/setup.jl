using Distributions
using LaTeXStrings
using LinearAlgebra
using Random: seed!

include("../setup/setup.jl")

seed!(16)

# ----------------
# Reservoir properties 
# ----------------

μ = 0.5 * 1e-3 / (3600.0 * 24.0)    # Viscosity (Pa⋅day)
ϕ = 0.30                            # Porosity
c = 2.0e-4 / 6895.0                 # Compressibility (Pa^-1)
u0 = 20 * 1.0e6                     # Initial pressure (Pa)

# ----------------
# Grid and boundary conditions
# ----------------

xmax = 1000.0                       # Dimension in x and y directions
tmax = 120.0

Δx_c = 12.5
Δx_f = 8.0
Δt_c = 4.0
Δt_f = 2.0

grid_c = Grid(xmax, tmax, Δx_c, Δt_c, ϕ, μ, c, u0)
grid_f = Grid(xmax, tmax, Δx_f, Δt_f, ϕ, μ, c, u0)

# ----------------
# Well parameters 
# ----------------

q_c = 30.0 / Δx_c^2                 # Producer rate, (m^3 / day) / m^3
q_f = 30.0 / Δx_f^2                 # Producer rate, (m^3 / day) / m^3

well_radius = 30.0

well_centres = [
    (150, 150), (150, 500), (150, 850),
    (500, 150), (500, 500), (500, 850),
    (850, 150), (850, 500), (850, 850)
]

well_rates_c = [
    (-q_c, 0, 0), (0, -q_c, 0), (-q_c, 0, 0),
    (0, -q_c, 0), (-q_c, 0, 0), (0, -q_c, 0),
    (-q_c, 0, 0), (0, -q_c, 0), (-q_c, 0, 0)
]

well_rates_f = [
    (-q_f, 0, 0), (0, -q_f, 0), (-q_f, 0, 0),
    (0, -q_f, 0), (-q_f, 0, 0), (0, -q_f, 0),
    (-q_f, 0, 0), (0, -q_f, 0), (-q_f, 0, 0)
]

well_change_times = [0, 40, 80] # First set on, second set on, all off

wells_c = [
    BumpWell(grid_c, centre..., well_radius, rates) 
    for (centre, rates) ∈ zip(well_centres, well_rates_c)
]

wells_f = [
    BumpWell(grid_f, centre..., well_radius, rates) 
    for (centre, rates) ∈ zip(well_centres, well_rates_f)
]

function build_Q(
    g::Grid,
    wells::AbstractVector{BumpWell},
    well_change_times::AbstractVector 
)

    Q_i = Int[]
    Q_j = Int[]
    Q_v = Float64[]

    # Find the well time period each time is within
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

    Q = sparse(Q_i, Q_j, Q_v, g.nx^2, g.nt+1)
    return Q

end

# Build matrix of forcing vectors
Q_c = build_Q(grid_c, wells_c, well_change_times)
Q_f = build_Q(grid_f, wells_f, well_change_times)

# ----------------
# Prior 
# ----------------

logp_mu = -13.5
σ_bounds = (0.25, 0.75)
l_bounds = (200, 400)

p = MaternField(grid_c, logp_mu, σ_bounds, l_bounds)

# ----------------
# Truth
# ----------------

true_field = MaternField(grid_f, logp_mu, σ_bounds, l_bounds)
θs_t = rand(true_field)
logps_t = transform(true_field, vec(θs_t))

us_t = solve(grid_f, vec(logps_t), Q_f)
us_t = reshape(us_t, grid_f.nx^2, grid_f.nt+1)

# ----------------
# Data
# ----------------

xs_obs = [c[1] for c ∈ well_centres]
ys_obs = [c[2] for c ∈ well_centres]
ts_obs = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]

# Operators that map between the output states and the data
B_f = build_observation_operator(grid_f, xs_obs, ys_obs)
B_c = build_observation_operator(grid_c, xs_obs, ys_obs)

ts_obs_inds_f = [findfirst(grid_f.ts .>= t) for t ∈ ts_obs]
ts_obs_inds_c = [findfirst(grid_c.ts .>= t) for t ∈ ts_obs]

n_obs = length(xs_obs) * length(ts_obs)

σ_ϵ = u0 * 0.01 # TODO: think about this (currently like this so the data are informative)
Γ = σ_ϵ^2 * Matrix(1.0I, n_obs, n_obs) 

us_obs = vcat([B_f * us_t[:, t] for t ∈ ts_obs_inds_f]...)
us_obs += rand(MvNormal(Γ))

# ----------------
# Model functions
# ----------------

function F(θs::AbstractVector)
    logps = transform(p, θs)
    return solve(grid_c, vec(logps), Q_c) # TODO: tidy
end

function F_r(θs::AbstractVector)
    logps = transform(p, θs)
    return solve(grid_c, vec(logps), Q_c, μ_u, V_r)
end

function G(us::AbstractVector)
    us = reshape(us, grid_c.nx^2, grid_c.nt+1)
    return vcat([B_c * us[:, t] for t ∈ ts_obs_inds_c]...)
end

# ----------------
# POD
# ----------------

us_samp = generate_pod_samples(p, 100)
μ_u, V_r = compute_pod_basis(grid_c, us_samp, 0.999)

function animate(us, grid, well_inds, fname)

    us = reshape(us, grid.nx, grid.nx, :)
    us ./= 1.0e6
    well_us = us[well_inds...,:]

    anim = @animate for i ∈ axes(us, 3)

        plot(
            heatmap(
                grid.xs, grid.xs, us[:, :, i]', 
                clims=extrema(us[2:end-1, 2:end-1, :]), 
                cmap=:turbo, 
                size=(500, 500),
                title="Reservoir pressure vs time",
                xlabel=L"x \, \textrm{(m)}",
                ylabel=L"y \, \textrm{(m)}"
            ),
            plot(
                grid.ts[1:i], well_us[1:i], 
                size=(500, 500), 
                xlims=(0, tmax),
                ylims=extrema(well_us),
                xlabel="Day",
                ylabel="Pressure (MPa)",
                title="Pressure in well at (500, 150)",
                legend=:none
            ),
            size=(1000, 400),
            margin=5Plots.mm
        )

    end

    gif(anim, "$fname.gif", fps=4)

end

TEST_POD = true

if TEST_POD

    θs_test = rand(p, 100)

    us_test = [@time F(θ) for θ ∈ eachcol(θs_test)]
    us_test_r = [@time F_r(θ) for θ ∈ eachcol(θs_test)]

    ys_test = hcat([G(u) for u ∈ us_test]...)
    ys_test_r = hcat([G(u) for u ∈ us_test_r]...)

    animate(us_test[1], grid_c, (41, 13), "darcy_flow_ex")
    animate(us_test_r[1], grid_c, (41, 13), "darcy_flow_ex_reduced")

end

animate(us_t, grid_f, (100, 100), "fine_grid")