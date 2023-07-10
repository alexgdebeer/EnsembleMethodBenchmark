using Distributions
using Interpolations
using LinearAlgebra
using PyCall
using Random
using SimIntensiveInference

# Random.seed!(1)

# TODO: extend things to production history
# Make a finer grid for the truth
# Save the full modelled temperatures somewhere

# ----------------
# Base model setup
# ----------------

@pyinclude "geothermal/model_functions.py"

xmax, nx = 1500.0, 20
ymax, ny = 75.0, 1
zmax, nz = 1500.0, 20

dx = xmax / nx
dz = zmax / nz

xs = collect(range(dx, xmax-dx, nx))
zs = collect(range(dz, zmax-dz, nz))

n_blocks = nx * nz

model_folder = "geothermal/models"
mesh_name = "gSQ$n_blocks"
model_name = "SQ$(n_blocks)"
model_path = "$(model_folder)/$(model_name)"

mass_cols = [9, 10]

py"build_base_model"(
    xmax, ymax, zmax, nx, ny, nz, 
    mesh_name, model_name, model_folder, mass_cols
)

mass_cells = py"get_mass_cells"(mesh_name, model_folder, mass_cols)

# ----------------
# Model functions 
# ----------------

global model_num = 1

function f(θs::AbstractVector)::Union{AbstractVector, Symbol}

    mass_rate = get_mass_rate(p, θs)
    logps = get_perms(p, θs)
    ps = reshape(10 .^ logps, n_blocks, 2)
    
    py"build_model"(model_folder, model_name, mass_rate, mass_cells, ps)
    py"run_model"(model_path)

    flag = py"run_info"(model_path)
    flag != "success" && @warn "Model failed. Flag: $(flag)."
    flag != "success" && return :failure 

    temps = py"get_quantity"(model_path, "fluid_temperature")
    return temps

end

function g(temps::Union{AbstractVector, Symbol})

    temps == :failure && return :failure
    temps = interpolate((xs, zs), reshape(temps, nx, nz), Gridded(Linear()))
    return [temps(x, z) for (x, z) ∈ zip(xs_o, zs_o)]

end

# ----------------
# Prior setup
# ----------------

cap_bnds = [-224.5, -75.5]
mass_rate_bnds = [10e-3, 15e-3]
logμ_reg = -14.0
logμ_cap = -16.0
k_reg = ARDExpSquaredKernel(0.5, 1500, 150)
k_cap = ARDExpSquaredKernel(0.25, 1500, 150)
ρ_xz = 0.8
level_width = 0.25

p = GeothermalPrior(
    cap_bnds,
    mass_rate_bnds, 
    logμ_reg, logμ_cap, 
    k_reg, k_cap, 
    ρ_xz, level_width, 
    xs, -zs
)

# ----------------
# Data generation 
# ----------------

# Generate the true set of parameters and outputs
θs_t = rand(p)
logps_t = get_perms(p, θs_t)
ps_t = 10 .^ logps_t
us_t = reshape(f(vec(θs_t)), nx, nz)

# Define the observation locations
x_locs = 250:200:1250
z_locs = 100:100:1400
n_obs = length(x_locs) * length(z_locs)

# Define the distribution of the observation noise
σ_ϵ_t = 2.0
Γ_ϵ = σ_ϵ_t^2 * Matrix(1.0I, n_obs, n_obs)
ϵ_dist = MvNormal(Γ_ϵ)

# Generate the data and add noise
us_t = interpolate((xs, zs), us_t, Gridded(Linear()))
xs_o = [x for x ∈ x_locs for _ ∈ z_locs]
zs_o = [z for _ ∈ x_locs for z ∈ z_locs]
us_o = [us_t(x, z) for (x, z) ∈ zip(xs_o, zs_o)] + rand(ϵ_dist)

# ----------------
# Likelihood setup 
# ----------------

σ_ϵ = 2.0
Γ_ϵ = σ_ϵ^2 * Matrix(1.0I, n_obs, n_obs)
L = MvNormal(us_o, Γ_ϵ)

# py"slice_plot"(model_folder, mesh_name, logps_t[1:n_blocks], cmap="turbo")