include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

Ne = 100
n_trials = 10

data_folder = "data/enrml"

fnames = [
    # "$(data_folder)/enrml_$Ne.h5", 
    # "$(data_folder)/enrml_boot_$Ne.h5", 
    # "$(data_folder)/enrml_boot_reg_$Ne.h5", 
    # "$(data_folder)/enrml_shuffle_$Ne.h5", 
    # "$(data_folder)/enrml_fisher_$Ne.h5",
    "$(data_folder)/enrml_inflation_$Ne.h5",
    "$(data_folder)/enrml_boot_inflation_$Ne.h5"
]

groups = [1:pr.Nu, pr.Nu+1, pr.Nu+2]

settings = [
    # (IdentityLocaliser(), IdentityInflator()),
    # (BootstrapLocaliser(type=:unregularised), IdentityInflator()),
    # (BootstrapLocaliser(type=:regularised), IdentityInflator()),
    # (ShuffleLocaliser(), IdentityInflator()),
    # (FisherLocaliser(), IdentityInflator()),
    (IdentityLocaliser(), AdaptiveInflator()),
    (BootstrapLocaliser(type=:unregularised), AdaptiveInflator())
]

for (fname, setting) ∈ zip(fnames, settings)
    
    results = Dict()

    for i ∈ 1:n_trials 
        
        θs, us, Fs, Gs, Ss, λs, en_ind = run_enrml(
            F, G, pr, d_obs, μ_e, C_e, Ne; 
            localiser=setting[1],
            inflator=setting[2]
        )

        μ_post = transform(pr, mean(θs[end], dims=2))
        μ_post = reshape(μ_post, grid_c.nx, grid_c.nx)
        
        σ_post = std(us[end], dims=2)
        σ_post = reshape(σ_post, grid_c.nx, grid_c.nx)

        results["θs_$i"] = θs[en_ind]
        results["us_$i"] = us[en_ind]
        results["Fs_$i"] = model_r.B_wells * Fs[en_ind]

        results["μ_post_$i"] = μ_post
        results["σ_post_$i"] = σ_post

        results["n_its_$i"] = length(θs)
        results["n_sims_$i"] = Ne * (length(θs) - 1)

    end

    save_results(results, fname)

end