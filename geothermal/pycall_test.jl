using PyCall
using SimIntensiveInference

@pyinclude "geothermal/model_functions.py"

model_folder = "geothermal/models"
mesh_name = "g2D"
base_model_name = "2D_base"

xmax, nx = 1000.0, 20
ymax, ny = 50.0, 1
zmax, nz = 1000.0, 20

dx = xmax / nx
dz = zmax / nz

xs = collect(range(dx, xmax-dx, nx))
zs = collect(range(dz, zmax-dz, nz))

py"build_base_model"(
    xmax, ymax, zmax, nx, ny, nz, 
    mesh_name, base_model_name, model_folder
)

# py"run_model"("$(model_folder)/$(model_name).json")
# flag = py"run_info"("$(model_folder)/$(model_name).yaml")

# Figure out what to do about clay cap 
# Adapt ensemble methods to account for model failures
# Extend things to production history

logμ_p = -14.0
σ_p = 0.5
γ_p = 200
k = ExpSquaredKernel(σ_p, γ_p)

# Coordinate arrays swapped due to Layermesh convention
p = GaussianPrior(logμ_p, k, zs, xs)

logps = vec(rand(p))
ps = 10 .^ logps

py"build_model"(model_folder, base_model_name, "test_model", ps)

@time py"run_model"("$(model_folder)/test_model.json")
flag = py"run_info"("$(model_folder)/test_model.yaml")

model_path = "$(model_folder)/test_model"

temps = py"get_temperatures"(model_path)

py"slice_plot"(model_folder, mesh_name, temps)