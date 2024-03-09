include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

Ne = 100
n_trials = 10

data_folder = "data/eks"

fname = "$(data_folder)/eks_$Ne.h5"

results = Dict()

for i ∈ 1:n_trials 
    
    θs, us, Fs, Gs, n_its = run_eks(F, G, pr, d_obs, μ_e, C_e, Ne)

    μ_post = transform(pr, mean(θs, dims=2))
    μ_post = reshape(μ_post, grid_c.nx, grid_c.nx)

    σ_post = std(us, dims=2)
    σ_post = reshape(σ_post, grid_c.nx, grid_c.nx)

    results["θs_$i"] = θs
    results["us_$i"] = us
    results["Fs_$i"] = model_r.B_wells * Fs
    results["ls_$i"] = [gauss_to_unif(ω_σ, σ_bounds...) for ω_σ ∈ θs[end-1, :]]

    results["μ_post_$i"] = μ_post
    results["σ_post_$i"] = σ_post

    results["n_its_$i"] = n_its
    results["n_sims_$i"] = Ne * (n_its - 1)

end

save_results(results, fname)