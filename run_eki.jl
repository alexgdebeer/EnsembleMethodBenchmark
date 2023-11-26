include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

Ne = 100
fname = "data/eki/eki_power.h5"

n_trials = 10
results = Dict()

for i ∈ 1:n_trials 
    
    ηs, θs, Fs, Gs = run_eki_dmc(
        F, G, pr, d_obs, μ_e, Γ_e, Ne; 
        localiser=PowerLocaliser()
    )

    results["ηs_$i"] = ηs[end]
    results["θs_$i"] = θs[end]
    results["Fs_$i"] = Fs[end]
    results["Gs_$i"] = Gs[end]

end

save_results(results, fname)