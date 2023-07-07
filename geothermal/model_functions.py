from enum import Enum
import json
import layermesh.mesh as lm
import pywaiwera
import yaml

class ExitFlag(Enum):

    success = 1
    max_its = 2
    aborted = 3
    unknown = 4


def build_base_model(
        xmax, ymax, zmax, nx, ny, nz,
        P_atm=1.0e+5, T_atm=20.0, P0=1.0e+5, T0=20.0, 
        permeability=1.0e-14, porosity=0.25,
        mesh_name="g2D", model_name="2D_base", model_folder="geothermal/models"
    ):

    dx = xmax / nx
    dy = ymax / ny
    dz = zmax / nz

    dxs = [dx] * nx
    dys = [dy] * ny
    dzs = [dz] * nz

    mesh = lm.mesh(rectangular=(dxs, dys, dzs))
    mesh.write(f"{model_folder}/{mesh_name}.h5")
    mesh.export(f"{model_folder}/{mesh_name}.exo")

    model = {
        "title" : "Simple 2D model",
        "eos" : {"name" : "we"}
    }

    model["mesh"] = {
        "filename" : f"{model_folder}/{mesh_name}.exo", 
        "thickness" : dy
    }

    model["gravity"] = 9.81

    model["rock"] = {"types" : [
        {
            "name" : f"{c.index}", 
            "permeability" : permeability, 
            "porosity" : porosity, 
            "cells" : [c.index]
        }
        for c in mesh.cell
    ]}

    inflow_cells = [mesh.column[n].cell[-1].index for n in [0, 9, 10, 11, 12, 13]]

    model["source"] = [{
        "component" : 1,
        "enthalpy" : 1.0e+7, 
        "rate" : 0.002,
        "cells" : inflow_cells
    }]

    model["initial"] = {
        "primary" : [P0, T0],
        "region" : 1
    }

    # Define atmosphere boundary condition
    model["boundaries"] = [{
        "primary": [P_atm, T_atm], 
        "region": 1,
        "faces": {
            "cells" : [c.index for c in mesh.surface_cells],
            "normal" : [0, 1]
        }
    }]

    model["time"] = {
        "step" : {
            "size" : 1.0e+6,
            "adapt" : {
                "on" : True,
                "method" : "iteration",
                "minimum" : 5, 
                "maximum" : 8
            }, 
            "maximum" : {"number" : 500},
            "method" : "beuler",
            "stop" : {"size" : {"maximum" : 1.0e+15}}
        }
    }

    model["output"] = {"filename" : f"{model_folder}/{model_name}.h5"}
    model["logfile"] = {"echo" : False}

    with open(f"{model_folder}/{model_name}.json", "w") as f:
        json.dump(model, f, indent=2, sort_keys=True)


def run_model(fname):

    env = pywaiwera.docker.DockerEnv()
    env.run_waiwera(fname, noupdate=True)


def run_info(log_fname):

    with open(log_fname, "r") as f:
        log = yaml.safe_load(f)

    for msg in log[:-20:-1]:

        if msg[:3] == ["info", "timestep", "end_time_reached"]:
            return ExitFlag.success
        
        elif msg[:3] == ["info", "timestep", "stop_size_maximum_reached"]:
            return ExitFlag.success

        elif msg[:3] == ["info", "timestep", "max_timesteps_reached"]:
            return ExitFlag.max_its

        elif msg[:3] == ["warn", "timestep", "aborted"]:
            return ExitFlag.aborted

    raise Exception("Unknown exit condition encountered. Check the log.")
    # return ExitFlag.unknown