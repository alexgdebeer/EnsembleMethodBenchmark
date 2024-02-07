include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

Ne = 100
n_trials = 10

data_folder = "data/eki"

fnames = [
    "$(data_folder)/eki_$Ne.h5", 
    # "$(data_folder)/eki_boot_$Ne.h5", 
    # "$(data_folder)/eki_boot_reg_$Ne.h5", 
    # "$(data_folder)/eki_shuffle_$Ne.h5", 
    # "$(data_folder)/eki_inflation_$Ne.h5",
    # "$(data_folder)/eki_boot_inflation_$Ne.h5",
]

settings = [
    (IdentityLocaliser(), IdentityInflator()),
    # (BootstrapLocaliser(type=:unregularised), IdentityInflator()),
    # (BootstrapLocaliser(type=:regularised), IdentityInflator()),
    # (ShuffleLocaliser(), IdentityInflator()),
    # (IdentityLocaliser(), AdaptiveInflator()),
    # (BootstrapLocaliser(type=:unregularised), AdaptiveInflator())
]

for (fname, setting) ∈ zip(fnames, settings)

    results = Dict()

    for i ∈ 1:n_trials 
        
        θs, us, Fs, Gs = run_eki_dmc(
            F, G, pr, d_obs, μ_e, C_e, Ne; 
            localiser=setting[1],
            inflator=setting[2]
        )

        μ_post = transform(pr, mean(θs[end], dims=2))
        μ_post = reshape(μ_post, grid_c.nx, grid_c.nx)
        
        σ_post = std(us[end], dims=2)
        σ_post = reshape(σ_post, grid_c.nx, grid_c.nx)

        results["θs_$i"] = θs[end]
        results["us_$i"] = us[end]
        results["Fs_$i"] = model_r.B_wells * Fs[end]

        results["μ_post_$i"] = μ_post
        results["σ_post_$i"] = σ_post
        
        results["n_its_$i"] = length(θs)
        results["n_sims_$i"] = Ne * (length(θs) - 1)

    end

    save_results(results, fname)

end