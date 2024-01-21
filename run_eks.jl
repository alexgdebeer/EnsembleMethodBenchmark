include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

Ne = 1000
n_trials = 10

data_folder = "data/eks"

# Could try uncorrected version too.
fname = "$(data_folder)/eks_$Ne.h5"

results = Dict()

for i ∈ 1:n_trials 
    
    θs, us, Fs, Gs = run_eks(F, G, pr, d_obs, μ_e, C_e, Ne)

    μ_post = transform(pr, mean(θs[end], dims=2))
    μ_post = reshape(μ_post, grid_c.nx, grid_c.nx)
    
    σ_post = std(us[end], dims=2)
    σ_post = reshape(σ_post, grid_c.nx, grid_c.nx)

    results["θs_$i"] = θs[end]
    results["us_$i"] = us[end]
    results["Gs_$i"] = Gs[end]

    results["μ_post_$i"] = μ_post
    results["σ_post_$i"] = σ_post
    
    results["n_its_$i"] = length(θs)
    results["n_sims_$i"] = Ne * length(θs)

end

save_results(results, fname)