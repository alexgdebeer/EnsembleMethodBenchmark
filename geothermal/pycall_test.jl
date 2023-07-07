using PyCall
using SimIntensiveInference

# TODO: Adapt ensemble methods to account for model failures
# TODO: Extend things to production history

@pyinclude "geothermal/model_functions.py"

model_folder = "geothermal/models"
mesh_name = "gSQ"
base_model_name = "SQ400"

xmax, nx = 1000.0, 20
ymax, ny = 50.0, 1
zmax, nz = 1000.0, 20

dx = xmax / nx
dz = zmax / nz

xs = collect(range(dx, xmax-dx, nx))
zs = collect(range(dz, zmax-dz, nz))

n_blocks = nx * nz

py"build_base_model"(
    xmax, ymax, zmax, nx, ny, nz, 
    mesh_name, base_model_name, model_folder
)

logμ_reg = -13.0
logμ_cap = -15.0
k_reg = ExpSquaredKernel(0.5, 200)
k_cap = ExpKernel(0.25, 200)
ρ_xz = 0.8

p = GeothermalPrior(logμ_reg, logμ_cap, k_reg, k_cap, ρ_xz, xs, -zs)
θs_t = rand(p)

global model_num = 1

function f(θs)

    logps = get_perms(p, θs)
    ps = reshape(10 .^ logps, n_blocks, 2)

    model_name = "SQ$(n_blocks)_$(model_num)"
    model_path = "$(model_folder)/$(model_name)"

    global model_num += 1
    
    py"build_model"(model_folder, base_model_name, model_name, ps)
    py"run_model"(model_path)
    flag = py"run_info"(model_path)
    println(flag)

    temps = py"get_temperatures"(model_path)
    return temps

end

temps = @time f(θs_t)

py"slice_plot"(model_folder, mesh_name, temps)