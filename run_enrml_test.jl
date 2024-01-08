include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

Ne = 100
    
θs, us, Fs, Gs, Ss, λs, en_ind = run_enrml(
    F, G, pr, d_obs, μ_e, Γ_e, Ne; 
    localiser=IdentityLocaliser(),
    inflator=AdaptiveInflator()
)