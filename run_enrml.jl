include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

Ne = 100
fname = "data/enrml/enrml_noloc.h5"

n_trials = 10
results = Dict()

for i ∈ 1:n_trials 
    
    ηs, θs, Fs, Gs, Ss, λs, en_ind = run_enrml(
        F, G, pr, d_obs, μ_e, Γ_e, Ne; 
        localiser=IdentityLocaliser()
    )

    results["ηs_$i"] = ηs[en_ind]
    results["θs_$i"] = θs[en_ind]
    results["Fs_$i"] = Fs[en_ind]
    results["Gs_$i"] = Gs[en_ind]

end

save_results(results, fname)