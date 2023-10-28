include("setup.jl")
include("InferenceAlgorithms/pcn.jl")

NF = 9 * grid_c.nt # TODO: put n_wells somewhere?
Ni = 500_000

β = 0.05
δ = 0.20

Nb = 100
Nc = 4

B_wells = blockdiag([grid_c.Bi for _ ∈ 1:grid_c.nt]...)

run_pcn(
    F_r, G, pr, y_obs, 
    μ_ε, L_e, 
    NF, Ni, Nb, Nc,
    β, δ, B_wells
)