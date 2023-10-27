using Distributions
using LinearAlgebra
using Random: seed!

include("DarcyFlow/DarcyFlow.jl")
include("plotting.jl")

seed!(16)

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

Δx_c = 12.5
Δx_f = 7.5
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

well_radius = 50.0
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

lnp_μ = -31
σ_bounds = (0.5, 1.25)
l_bounds = (100, 1000)

pr = MaternField(grid_c, lnp_μ, σ_bounds, l_bounds)

# ----------------
# Truth
# ----------------

true_field = MaternField(grid_f, lnp_μ, σ_bounds, l_bounds)
η_t = rand(true_field)
θ_t = transform(true_field, η_t)

u_t = solve(grid_f, θ_t, Q_f)

# ----------------
# Data
# ----------------

σ_ϵ = u0 * 0.01
Γ_ϵ = diagm(fill(σ_ϵ^2, grid_f.ny))
Γ_ϵ_inv = spdiagm(fill(σ_ϵ^-2, grid_f.ny))

y_obs = grid_f.B * u_t
y_obs += rand(MvNormal(Γ_ϵ))

# ----------------
# Model functions
# ----------------

function F(η::AbstractVector)
    θ = transform(pr, η)
    return solve(grid_c, θ, Q_c)
end

function F_r(η::AbstractVector)
    θ = transform(pr, η)
    return solve(grid_c, θ, Q_c, μ_u, V_r)
end

function G(us::AbstractVector)
    return grid_c.B * us
end

# ----------------
# POD
# ----------------

# Generate POD basis 
# μ_u, V_r, μ_ε, Γ_ε = generate_pod_data(grid_c, F, G, pr, 100, 0.999, "pod$(grid_c.nx)")
μ_u, V_r, μ_ε, Γ_ε = read_pod_data("pod$(grid_c.nx)")

Γ_e = Hermitian(Γ_ϵ + Γ_ε)
Γ_e_inv = Hermitian(inv(Γ_e))
L_e = cholesky(Γ_e_inv).U

# if TEST_POD

#     θs_test = rand(pr, 20)

#     us_test = [@time F(θ) for θ ∈ eachcol(θs_test)]
#     us_test_r = [@time F_r(θ) for θ ∈ eachcol(θs_test)]

#     ys_test = hcat([G(u) for u ∈ us_test]...)
#     ys_test_r = hcat([G(u) for u ∈ us_test_r]...)

#     animate(us_test[1], grid_c, (41, 13), "plots/animations/darcy_flow_ex")
#     animate(us_test_r[1], grid_c, (41, 13), "plots/animations/darcy_flow_ex_reduced")

# end

# animate(u_t, grid_f, (8, 8), "plots/animations/test")