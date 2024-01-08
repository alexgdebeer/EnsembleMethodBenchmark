using HDF5

include("setup.jl")

fname_mcmc = "data/pcn/pcn.h5"

laplace_folder = "data/laplace"
enrml_folder = "data/enrml"
eki_folder = "data/eki"

fnames = [

    "$(laplace_folder)/laplace_100.h5",
    "$(laplace_folder)/laplace_1000.h5",

    "$(enrml_folder)/enrml_100.h5", 
    "$(enrml_folder)/enrml_1000.h5", 
    "$(enrml_folder)/enrml_boot_100.h5", 
    "$(enrml_folder)/enrml_boot_reg_100.h5", 
    "$(enrml_folder)/enrml_shuffle_100.h5", 
    # "$(enrml_folder)/enrml_fisher_100.h5",
    "$(enrml_folder)/enrml_inflation_100.h5",
    # "$(enrml_folder)/enrml_fisher_inflation_100.h5",

    "$(eki_folder)/eki_100.h5",
    "$(eki_folder)/eki_1000.h5",
    "$(eki_folder)/eki_boot_100.h5",
    "$(eki_folder)/eki_boot_reg_100.h5",
    "$(eki_folder)/eki_shuffle_100.h5",
    # "$(eki_folder)/eki_sec_100.h5",
    # "$(eki_folder)/eki_fisher_100.h5",
    "$(eki_folder)/eki_inflation_100.h5",
    # "$(eki_folder)/eki_fisher_inflation_100.h5",
    "$(eki_folder)/eki_boot_inflation_100.h5"

]

function get_results_mcmc(fname)

    f = h5open(fname, "r")
    μ = f["mean"][:, :]
    σ = f["stds"][:, :]

    return μ, σ
    
end

function get_results_trial(fname, i_trial)

    f = h5open(fname, "r")
    μ = read(f["μ_post_$(i_trial)"])
    σ = read(f["σ_post_$(i_trial)"])
    θs = read(f["θs_$(i_trial)"])
    μ = reshape(transform(pr, mean(θs, dims=2)), 80, 80)
    sim = read(f["n_sims_$(i_trial)"])
    close(f)

    return μ, σ, sim

end

function get_results_trials(fname, n_trials)

    μs, σs, sims = [], [], []
    for i ∈ 1:n_trials
        μ_i, σ_i, sims_i = get_results_trial(fname, i)
        push!(μs, μ_i)
        push!(σs, σ_i)
        push!(sims, sims_i)
    end
    return μs, σs, sims

end

function compute_ϵ_μ(μ_mcmc, μ_pri, μ_approx)
    ϵ_μ = norm(μ_approx - μ_mcmc) / norm(μ_mcmc - μ_pri)
    return ϵ_μ
end

function compute_ϵ_σ(σ_mcmc, σ_approx)
    return norm(σ_mcmc - σ_approx) / norm(σ_mcmc)
end

function compute_metrics(
    fname, 
    n_trials, 
    μ_pri, 
    μ_mcmc, 
    σ_mcmc
)

    μs, σs, sims = get_results_trials(fname, n_trials)

    ϵs_μ = [compute_ϵ_μ(μ_mcmc, μ_pri, μ) for μ ∈ μs]
    ϵs_σ = [compute_ϵ_σ(σ_mcmc, σ) for σ ∈ σs]

    return mean(ϵs_μ), mean(ϵs_σ), mean(sims)

end

n_trials = 10

μ_pri = transform(pr, zeros(pr.Nθ))
μ_pri = reshape(μ_pri, grid_c.nx, grid_c.nx)
μ_mcmc, σ_mcmc = get_results_mcmc(fname_mcmc)

for fname ∈ fnames

    ϵ_μ, ϵ_σ, n_sims = compute_metrics(
        fname, n_trials, μ_pri, μ_mcmc, σ_mcmc
    )

    println(fname)
    println("ϵ_μ: $(ϵ_μ)")
    println("ϵ_σ: $(ϵ_σ)")
    println("Mean simulations: $(n_sims)")

end