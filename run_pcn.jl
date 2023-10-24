include("setup.jl")
include("InferenceAlgorithms/pcn.jl")

NF = grid_c.nx^2 * grid_c.nt
Ni = 500_000
Nc = 1

β = 0.05
δ = 0.5

ηs, Fs, Gs, τs = run_pcn(F_r, G, pr, y_obs, μ_ε, L_e, NF, Ni, Nc, β, δ, save_increment=100)