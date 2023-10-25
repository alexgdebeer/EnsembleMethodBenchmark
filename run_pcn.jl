include("setup.jl")
include("InferenceAlgorithms/pcn.jl")

NF = grid_c.nx^2 * grid_c.nt
Ni = 500_000

β = 0.05
δ = 0.5

chain_num = 1
Nb = 100

run_pcn(
    F_r, G, pr, y_obs, 
    μ_ε, L_e, 
    NF, Ni, Nb,
    β, δ, 
    chain_num
)