using Distributions
using HDF5
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
p0 = 20 * 1.0e6                     # Initial pressure (Pa)

# ----------------
# Grid
# ----------------

xmax = 1000.0
tmax = 120.0

Δx_c = 12.5
Δx_f = 7.5
Δt_c = 4.0
Δt_f = 2.0

n_wells = 9
well_centres = [
    (150, 150), (150, 500), (150, 850),
    (500, 150), (500, 500), (500, 850),
    (850, 150), (850, 500), (850, 850)
]

x_obs = [c[1] for c ∈ well_centres]
y_obs = [c[2] for c ∈ well_centres]
t_obs = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]

grid_c = Grid(xmax, tmax, Δx_c, Δt_c)
grid_f = Grid(xmax, tmax, Δx_f, Δt_f)

# ----------------
# Well parameters 
# ----------------

q_c = 50.0 / Δx_c^2                 # Extraction rate, (m^3 / day) / m^3
q_f = 50.0 / Δx_f^2                 # Extraction rate, (m^3 / day) / m^3

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

model_f = Model(grid_f, ϕ, μ, c, p0, wells_f, well_change_times, x_obs, y_obs, t_obs)
model_c = Model(grid_c, ϕ, μ, c, p0, wells_c, well_change_times, x_obs, y_obs, t_obs)

# ----------------
# Truth
# ----------------

lnk_μ = -31
σ_bounds = (0.5, 1.25)
l_bounds = (200, 1000)

true_field = MaternField(grid_f, lnk_μ, σ_bounds, l_bounds)

θ_t = rand(true_field)
u_t = transform(true_field, θ_t)
p_t = solve(grid_f, model_f, u_t)

# ----------------
# Data
# ----------------

σ_ϵ = p0 * 0.01
C_ϵ = diagm(fill(σ_ϵ^2, model_f.ny))
C_ϵ_inv = spdiagm(fill(σ_ϵ^-2, model_f.ny))

d_obs = model_f.B * p_t
d_obs += rand(MvNormal(C_ϵ))

# ----------------
# Parameter POD 
# ----------------

Nθ = grid_c.nx^2 + 2
# L_θ, Λ_θr, V_θr = generate_pod_params(Nθ, "pod/params_$(grid_c.nx)", N_r=1000)
L_θ, Λ_θr, V_θr = read_pod_params_data("pod/params_$(grid_c.nx)")

pr = ReducedMaternField(grid_c, lnk_μ, σ_bounds, l_bounds, L_θ, Λ_θr, V_θr)

# ----------------
# State POD
# ----------------

# Generate POD basis 
# μ_pi, V_ri, μ_ε, Γ_ε = generate_pod_data(grid_c, model_c, pr, 100, 0.999, "pod/grid_$(grid_c.nx)")
μ_pi, V_ri, μ_ε, C_ε = read_pod_data("pod/grid_$(grid_c.nx)") # TODO: tidy up

μ_e = μ_ε .+ 0.0
C_e = Hermitian(C_ϵ + C_ε)
C_e_inv = Hermitian(inv(C_e))
# L_e = cholesky(C_e_inv).U

model_r = ReducedOrderModel(
    grid_c, ϕ, μ, c, p0, wells_c, well_change_times,
    x_obs, y_obs, t_obs, μ_pi, V_ri, μ_e, C_e
)

# ----------------
# Model functions
# ----------------

F(u::AbstractVector) = solve(grid_c, model_r, u)
G(p::AbstractVector) = model_c.B * p