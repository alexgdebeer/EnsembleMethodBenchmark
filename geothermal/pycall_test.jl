using PyCall

@pyinclude "geothermal/model_functions.py"

model_folder = "geothermal/models"
mesh_name = "g2D"
model_name = "2D_base"

# Define geometry parameters
xmax, nx = 1000.0, 20
ymax, ny = 50.0, 1
zmax, nz = 1000.0, 20

py"build_base_model"(xmax, ymax, zmax, nx, ny, nz)
py"run_model"("$(model_folder)/$(model_name).json")
flag = py"run_info"("$(model_folder)/$(model_name).yaml")