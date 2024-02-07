using Distributions
using HDF5
using LinearAlgebra
using Random: seed!

include("DarcyFlow/DarcyFlow.jl")
include("plotting.jl")

seed!(16)

FILE_TRUTH = "data/truth.h5"

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
tmax = 160.0

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
well_change_times = [0, 40, 80, 120]

well_rates_c = [
    (-q_c, 0, 0, -0.5q_c), (0, -q_c, 0, -0.5q_c), (-q_c, 0, 0, -0.5q_c),
    (0, -q_c, 0, -0.5q_c), (-q_c, 0, 0, -0.5q_c), (0, -q_c, 0, -0.5q_c),
    (-q_c, 0, 0, -0.5q_c), (0, -q_c, 0, -0.5q_c), (-q_c, 0, 0, -0.5q_c)
]

well_rates_f = [
    (-q_f, 0, 0, -0.5q_f), (0, -q_f, 0, -0.5q_f), (-q_f, 0, 0, -0.5q_f),
    (0, -q_f, 0, -0.5q_f), (-q_f, 0, 0, -0.5q_f), (0, -q_f, 0, -0.5q_f),
    (-q_f, 0, 0, -0.5q_f), (0, -q_f, 0, -0.5q_f), (-q_f, 0, 0, -0.5q_f)
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

function generate_truth(
    g::Grid,
    m::Model,
    μ::Real,
    σ_bounds::Tuple,
    l_bounds::Tuple
)

    true_field = MaternField(g, μ, σ_bounds, l_bounds)
    θ_t = vec(rand(true_field))
    u_t = transform(true_field, θ_t)
    F_t = solve(g, m, u_t)
    G_t = m.B * F_t

    h5write(FILE_TRUTH, "θ_t", θ_t)
    h5write(FILE_TRUTH, "u_t", u_t)
    h5write(FILE_TRUTH, "F_t", F_t)
    h5write(FILE_TRUTH, "G_t", G_t)

    return θ_t, u_t, F_t, G_t

end

function generate_obs(
    G_t::AbstractVector, 
    C_ϵ::AbstractMatrix
)

    d_obs = G_t + rand(MvNormal(C_ϵ)) 
    h5write(FILE_TRUTH, "d_obs", d_obs)

    return d_obs

end

function read_truth()

    f = h5open(FILE_TRUTH)
    θ_t = f["θ_t"][:]
    u_t = f["u_t"][:]
    F_t = f["F_t"][:]
    G_t = f["G_t"][:]
    close(f)

    return θ_t, u_t, F_t, G_t

end

function read_obs()

    f = h5open(FILE_TRUTH)
    d_obs = f["d_obs"][:]
    close(f)

    return d_obs

end

# ----------------
# Prior and error distribution
# ----------------

lnk_μ = -31
σ_bounds = (0.5, 1.25)
l_bounds = (200, 1000)

pr = MaternField(grid_c, lnk_μ, σ_bounds, l_bounds)

σ_ϵ = p0 * 0.01
C_ϵ = diagm(fill(σ_ϵ^2, model_f.ny))

C_ϵ_inv = spdiagm(fill(σ_ϵ^-2, model_f.ny))

# ----------------
# Truth and observations
# ----------------

# θ_t, u_t, F_t, G_t = generate_truth(grid_f, model_f, lnk_μ, σ_bounds, l_bounds)
θ_t, u_t, F_t, G_t = read_truth()

# d_obs = generate_obs(G_t, C_ϵ)
d_obs = read_obs()

# ----------------
# POD
# ----------------

# Generate POD basis 
# μ_pi, V_ri, μ_ε, C_ε = generate_pod_data(grid_c, model_c, pr, 100, 0.999, "pod/grid_$(grid_c.nx)")
μ_pi, V_ri, μ_ε, C_ε = read_pod_data("pod/grid_$(grid_c.nx)")

μ_e = μ_ε .+ 0.0
C_e = Hermitian(C_ϵ + C_ε)
C_e_inv = Hermitian(inv(C_e))
L_e = cholesky(C_e_inv).U

model_r = ReducedOrderModel(
    grid_c, ϕ, μ, c, p0, wells_c, well_change_times,
    x_obs, y_obs, t_obs, μ_pi, V_ri, μ_e, C_e
)

# ----------------
# Model functions
# ----------------

F(u::AbstractVector) = solve(grid_c, model_r, u)
G(p::AbstractVector) = model_c.B * p

# if TEST_POD

#     θs_test = rand(pr, 20)

#     us_test = [@time F(θ) for θ ∈ eachcol(θs_test)]
#     us_test_r = [@time F_r(θ) for θ ∈ eachcol(θs_test)]

#     ys_test = hcat([G(u) for u ∈ us_test]...)
#     ys_test_r = hcat([G(u) for u ∈ us_test_r]...)

#     animate(us_test[1], grid_c, (41, 13), "plots/animations/darcy_flow_ex")
#     animate(us_test_r[1], grid_c, (41, 13), "plots/animations/darcy_flow_ex_reduced")

# end

# animate(u_t, grid_f, (50, 50), "plots/animations/test")

# # # Stuff to export for plotting (TODO: clean up...)
# well_ps = hcat(fill(p0, 9), reshape(model_f.B_wells * F_t, 9, :))
# well_ts = [0.0, grid_f.ts...]

# well_ps_obs = reshape(d_obs, 9, :)
# well_ts_obs = t_obs
# u_t = reshape(u_t, grid_f.nx, grid_f.nx)

# fname = "data/setup_alt.h5"

# h5write(fname, "well_ps", well_ps)
# h5write(fname, "well_ts", well_ts)
# h5write(fname, "well_ps_obs", well_ps_obs)
# h5write(fname, "well_ts_obs", well_ts_obs)
# h5write(fname, "u_t", u_t)
# h5write(fname, "xs", collect(grid_f.xs))
# h5write(fname, "well_centres", well_centres)