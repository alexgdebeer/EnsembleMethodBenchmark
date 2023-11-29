using HDF5

"""Comparison between the results of various algorithms and MCMC."""

fname_mcmc = "data/pcn/pcn.h5"

fname_eki_noloc = "data/eki/eki_noloc.h5"
fname_eki_fisher = "data/eki/eki_fisher.h5"
fname_eki_boot = "data/eki/eki_boot.h5"
fname_eki_shuffle = "data/eki/eki_shuffle.h5"
fname_eki_power = "data/eki/eki_power.h5"

fname_enrml_noloc = "data/enrml/enrml_noloc.h5"
fname_enrml_fisher = "data/enrml/enrml_fisher.h5"
fname_enrml_boot = "data/enrml/enrml_boot.h5"

function get_post_mean(ηs::AbstractMatrix)
    μ_η = mean(ηs, dims=2)
    μ_post = transform(pr, μ_η)
    return reshape(μ_post, grid_c.nx, grid_c.nx)
end

# Standard deviations or variances?
function get_post_stds(θs::AbstractMatrix)
    σ_post = std(θs, dims=2)
    return reshape(σ_post, grid_c.nx, grid_c.nx)
end

function get_results_mcmc(fname)

    f = h5open(fname, "r")
    μ = f["μ"][:, :]
    σ = f["σ"][:, :]

    return μ, σ
    
end

function get_results_trial(fname, i_trial)

    f = h5open(fname, "r")
    ηs = f["ηs_$(i_trial)"][:, :]
    θs = f["θs_$(i_trial)"][:, :]
    close(f)

    return ηs, θs

end

function get_results_trials(fname, n_trials)

    ηs = []
    θs = []
    for i ∈ 1:n_trials
        ηs_i, θs_i = get_results_trial(fname, i)
        push!(ηs, ηs_i)
        push!(θs, θs_i)
    end
    return ηs, θs

end

function compute_ϵ_μ(μ_mcmc, μ_pri, μ_approx)
    # Frobenius norm 
    ϵ_μ = norm(μ_approx - μ_mcmc) / norm(μ_mcmc - μ_pri)
    return ϵ_μ
end

function compute_ϵ_σ(σ_mcmc, σ_approx)
    return norm(σ_mcmc - σ_approx) / norm(σ_approx)
end

function compute_similarity_metrics(
    fname, 
    n_trials, 
    μ_pri, 
    μ_mcmc, 
    σ_mcmc
)

    ηs, θs = get_results_trials(fname, n_trials)
    μs = [get_post_mean(ηs_i) for ηs_i ∈ ηs]
    σs = [get_post_stds(θs_i) for θs_i ∈ θs]

    ϵs_μ = [compute_ϵ_μ(μ_mcmc, μ_pri, μ) for μ ∈ μs]
    ϵs_σ = [compute_ϵ_σ(σ_mcmc, σ) for σ ∈ σs]

    return mean(ϵs_μ), mean(ϵs_σ)

end

n_trials = 10

μ_pri = transform(pr, zeros(pr.Nη))
μ_pri = reshape(μ_pri, grid_c.nx, grid_c.nx)
μ_mcmc, σ_mcmc = get_results_mcmc(fname_mcmc)

ϵ_μ_eki_noloc, ϵ_σ_eki_noloc = compute_similarity_metrics(
    fname_eki_noloc, n_trials, μ_pri, μ_mcmc, σ_mcmc
)

ϵ_μ_eki_fisher, ϵ_σ_eki_fisher = compute_similarity_metrics(
    fname_eki_fisher, n_trials, μ_pri, μ_mcmc, σ_mcmc
)

ϵ_μ_eki_boot, ϵ_σ_eki_boot = compute_similarity_metrics(
    fname_eki_boot, n_trials, μ_pri, μ_mcmc, σ_mcmc
)

ϵ_μ_eki_shuffle, ϵ_σ_eki_shuffle = compute_similarity_metrics(
    fname_eki_shuffle, n_trials, μ_pri, μ_mcmc, σ_mcmc
)

ϵ_μ_eki_power, ϵ_σ_eki_power = compute_similarity_metrics(
    fname_eki_power, n_trials, μ_pri, μ_mcmc, σ_mcmc
)

ϵ_μ_enrml_noloc, ϵ_σ_enrml_noloc = compute_similarity_metrics(
    fname_enrml_noloc, n_trials, μ_pri, μ_mcmc, σ_mcmc
)

ϵ_μ_enrml_fisher, ϵ_σ_enrml_fisher = compute_similarity_metrics(
    fname_enrml_fisher, n_trials, μ_pri, μ_mcmc, σ_mcmc
)

ϵ_μ_enrml_boot, ϵ_σ_enrml_boot = compute_similarity_metrics(
    fname_enrml_boot, n_trials, μ_pri, μ_mcmc, σ_mcmc
)