"""Runs the ES on the linear model."""

include("linear_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(LinearModel.μ_π, LinearModel.Γ_π)

const es = true
const es_mda = true

const N_e = 10_000

if es

    θs, ys = SimIntensiveInference.run_es(
        LinearModel.f, LinearModel.g, π,  
        LinearModel.YS_O, 
        LinearModel.σ_ϵ, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs[end]), 
        LinearModel.θ1S, LinearModel.θ2S, 
        LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
        "Linear Model: ES Posterior",
        "$(LinearModel.PLOTS_DIR)/es/es_posterior.pdf";
        θs_t=LinearModel.θS_T,
        caption="Ensemble size: $N_e."
    )

    Plotting.plot_lm_posterior_predictions(
        LinearModel.TS, ys[end], LinearModel.YS_T, 
        LinearModel.TS_O, LinearModel.YS_O, 
        "Linear Model: ES Posterior Predictions",
        "$(LinearModel.PLOTS_DIR)/es/es_posterior_predictions.pdf"
    )

end

if es_mda

    const αs = [16.0 for _ ∈ 1:16]

    θs, ys = SimIntensiveInference.run_es_mda(
        LinearModel.f, LinearModel.g, π,  
        LinearModel.YS_O, 
        LinearModel.σ_ϵ, αs, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs[end]), 
        LinearModel.θ1S, LinearModel.θ2S, 
        LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
        "Linear Model: ES-MDA Posterior",
        "$(LinearModel.PLOTS_DIR)/es/es_mda_posterior.pdf";
        θs_t=LinearModel.θS_T,
        caption="Ensemble size: $N_e."
    )

    Plotting.plot_lm_posterior_predictions(
        LinearModel.TS, ys[end], LinearModel.YS_T, 
        LinearModel.TS_O, LinearModel.YS_O, 
        "Linear Model: ES-MDA Posterior Predictions",
        "$(LinearModel.PLOTS_DIR)/es/es_mda_posterior_predictions.pdf"
    )

end
