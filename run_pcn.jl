include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

NF = grid_c.nx^2 * grid_c.nt
Ni = 2_000_000

β = 0.025
δ = 0.20

Nb = 100
Nc = 5

run_pcn(
    F, G, pr, d_obs, 
    μ_ε, L_e, 
    NF, Ni, Nb, Nc,
    β, δ
)