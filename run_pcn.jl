include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

NF = n_wells * grid_c.nt
Ni = 500_000

β = 0.025
δ = 0.20

Nb = 100
Nc = 4

run_pcn(
    F, G, pr, d_obs, 
    μ_ε, L_e, 
    NF, Ni, Nb, Nc,
    β, δ, model_r.B_wells
)