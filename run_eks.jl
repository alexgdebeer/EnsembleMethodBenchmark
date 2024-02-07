include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

Ne = 100
n_trials = 10

data_folder = "data/eks"

fname = "$(data_folder)/eks_$Ne.h5"

results = Dict()

for i ∈ 1:n_trials 
    
θs, us, Fs, Gs = run_eks(F, G, pr, d_obs, μ_e, C_e, Ne; t_stop=5)

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