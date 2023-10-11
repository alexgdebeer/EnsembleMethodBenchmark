using Distributions
using LinearAlgebra
using Random: seed!

include("DarcyFlow/DarcyFlow.jl")
include("plotting.jl")

seed!(0)

# ----------------
# Reservoir properties 
# ----------------

μ = 0.5 * 1e-3 / (3600.0 * 24.0)    # Viscosity (Pa⋅day)
ϕ = 0.30                            # Porosity
c = 2.0e-4 / 6895.0                 # Compressibility (Pa^-1)
u0 = 20 * 1.0e6                     # Initial pressure (Pa)

# ----------------
# Grid
# ----------------

xmax = 1000.0
tmax = 120.0

Δx_c = 25.0
Δx_f = 25.0
Δt_c = 4.0
Δt_f = 2.0

well_centres = [
    (150, 150), (150, 500), (150, 850),
    (500, 150), (500, 500), (500, 850),
    (850, 150), (850, 500), (850, 850)
]

x_obs = [c[1] for c ∈ well_centres]
y_obs = [c[2] for c ∈ well_centres]
t_obs = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]

grid_c = Grid(xmax, tmax, Δx_c, Δt_c, x_obs, y_obs, t_obs, ϕ, μ, c, u0)
grid_f = Grid(xmax, tmax, Δx_f, Δt_f, x_obs, y_obs, t_obs, ϕ, μ, c, u0)

# ----------------
# Well parameters 
# ----------------

q_c = 30.0 / Δx_c^2                 # Extraction rate, (m^3 / day) / m^3
q_f = 30.0 / Δx_f^2                 # Extraction rate, (m^3 / day) / m^3

well_radius = 30.0
well_change_times = [0, 40, 80]

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
    Well(grid_c, centre..., well_radius, rates) 
    for (centre, rates) ∈ zip(well_centres, well_rates_c)
]

wells_f = [
    Well(grid_f, centre..., well_radius, rates) 
    for (centre, rates) ∈ zip(well_centres, well_rates_f)
]

Q_c = build_Q(grid_c, wells_c, well_change_times)
Q_f = build_Q(grid_f, wells_f, well_change_times)

# ----------------
# Prior 
# ----------------

lnp_mu = -31
σ = 1.0
l = 300
ν = 1.0

pr = MaternFieldKL(grid_c, lnp_mu, σ, l, ν)

# ----------------
# Truth
# ----------------

true_field = MaternFieldKL(grid_f, lnp_mu, σ, l, ν)
lnps_t = rand(true_field)

us_t = solve(grid_f, vec(lnps_t), Q_f)

# ----------------
# Data
# ----------------

σ_ϵ = u0 * 0.01 # TODO: think about this (currently like this so the data are informative)
Γ_ϵ = diagm(fill(σ_ϵ^2, grid_f.ny))
Γ_ϵ_inv = spdiagm(fill(σ_ϵ^-2, grid_f.ny))

y_obs = grid_f.B * us_t
y_obs += rand(MvNormal(Γ_ϵ))

# ----------------
# Model functions
# ----------------

function F(lnps::AbstractVector)
    return solve(grid_c, lnps, Q_c)
end

function F_r(lnps::AbstractVector)
    return solve(grid_c, lnps, Q_c, μ_u, V_r)
end

function G(us::AbstractVector)
    return grid_c.B * us
end

# ----------------
# POD
# ----------------

# us_samp = generate_pod_samples(p, 100)
# μ_u, V_r = compute_pod_basis(grid_c, us_samp, 0.999)

TEST_POD = false

if TEST_POD

    θs_test = rand(p, 100)

    us_test = [@time F(θ) for θ ∈ eachcol(θs_test)]
    us_test_r = [@time F_r(θ) for θ ∈ eachcol(θs_test)]

    ys_test = hcat([G(u) for u ∈ us_test]...)
    ys_test_r = hcat([G(u) for u ∈ us_test_r]...)

    animate(us_test[1], grid_c, (41, 13), "plots/animations/darcy_flow_ex")
    animate(us_test_r[1], grid_c, (41, 13), "plots/animations/darcy_flow_ex_reduced")

end

# animate(us_t, grid_f, (30, 30), "plots/animations/fine_grid")