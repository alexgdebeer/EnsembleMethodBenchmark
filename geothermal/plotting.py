import h5py
import layermesh.mesh as lm

mesh_name = "mesh_2d"
model_name = "model_2d"

mesh = lm.mesh(f"{mesh_name}.h5")
results = h5py.File(f"{model_name}.h5", "r")
index = results["cell_index"][:,0]

T = results["cell_fields"]["fluid_temperature"][-1][index]
P = results["cell_fields"]["fluid_pressure"][-1][index]

mesh.slice_plot(
    value = T,
    value_label = "Temperature",
    value_unit = "deg C",
    colourmap = "coolwarm"
)