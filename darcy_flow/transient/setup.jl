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

xmin, xmax = 0.0, 1000.0
ymin, ymax = 0.0, 1000.0
tmax = 120.0

Δx_c, Δy_c = 12.5, 12.5 # 80 * 80
Δx_f, Δy_f = 8.0, 8.0   # 120 * 120
Δt_f = 2.0
Δt_c = 4.0

well_change_times = [0, 40, 80] # TODO: not sure if this should be stored in the grid or not

q_c = 30.0 / (Δx_c * Δy_c)          # Producer rate, (m^3 / day) / m^3
q_f = 30.0 / (Δx_f * Δy_f)          # Producer rate, (m^3 / day) / m^3

grid_c = TransientGrid(xmin:Δx_c:xmax, ymin:Δy_c:ymax, tmax, Δt_c, well_change_times, μ, ϕ, c)
grid_f = TransientGrid(xmin:Δx_f:xmax, ymin:Δy_f:ymax, tmax, Δt_f, well_change_times, μ, ϕ, c)

bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => BoundaryCondition(:y0, :neumann, (x, y) -> 0.0), 
    :y1 => BoundaryCondition(:y1, :neumann, (x, y) -> 0.0),
    :t0 => BoundaryCondition(:t0, :initial, (x, y) -> u0)
)

# ----------------
# Well parameters 
# ----------------

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

wells_c = [
    BumpWell(grid_c, cs..., well_radius, qs) 
    for (cs, qs) ∈ zip(well_centres, well_rates_c)
]

wells_f = [
    BumpWell(grid_f, cs..., well_radius, qs) 
    for (cs, qs) ∈ zip(well_centres, well_rates_f)
]

Q_c = sum([w.Q for w ∈ wells_c])
Q_f = sum([w.Q for w ∈ wells_f])

# ----------------
# Prior 
# ----------------

logp_mu = -13.5
σ_bounds = (0.25, 0.75)
l_bounds = (100, 400)

p = MaternField(grid_c, logp_mu, σ_bounds, l_bounds)

# ----------------
# Truth
# ----------------

true_field = MaternField(grid_f, logp_mu, σ_bounds, l_bounds)
θs_t = rand(true_field)
logps_t = transform(true_field, vec(θs_t))

# logps_t = -0*ones(grid_f.nx, grid_f.ny)

us_t = solve(grid_f, logps_t, bcs, Q_f)
us_t = reshape(us_t, grid_f.nu, grid_f.nt+1)

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
    return solve(grid_c, logps, bcs, Q_c)
end

function F_r(θs::AbstractVector)
    logps = transform(p, θs)
    return solve(grid_c, logps, bcs, Q_c, μ_u, V_r)
end

function G(us::AbstractVector)
    us = reshape(us, grid_c.nu, grid_c.nt+1)
    return vcat([B_c * us[:, t] for t ∈ ts_obs_inds_c]...)
end

# ----------------
# POD
# ----------------

# us_samp = generate_pod_samples(p, 100)
# μ_u, V_r = compute_pod_basis(grid_c, us_samp, 0.9995)

function animate(us, grid, well_inds, fname)

    us = reshape(us, grid.nx, grid.ny, :)
    us ./= 1.0e6
    well_us = us[well_inds...,:]

    anim = @animate for i ∈ axes(us, 3)

        plot(
            heatmap(
                grid.xs, grid.ys, us[:, :, i]', 
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

TEST_POD = false

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