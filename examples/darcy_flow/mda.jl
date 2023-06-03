using SimIntensiveInference

include("problem_setup.jl")

# Define inflation factors
αs = [16.0 for _ ∈ 1:16]
N_e = 1000

logps, us = SimIntensiveInference.run_es_mda(f, g, π, L, αs, N_e)