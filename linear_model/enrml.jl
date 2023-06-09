"""Runs EnRML on the linear model."""

using LaTeXStrings

include("linear_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(LinearModel.μ_π, LinearModel.Γ_π)

const batch_enrml = true
const lm_enrml = true

const i_max = 16
const N_e = 10_000

if batch_enrml

    const β_0 = 0.5

    θs, ys, Ss, βs = SimIntensiveInference.run_batch_enrml(
        LinearModel.f, LinearModel.g, π,  
        LinearModel.YS_O, LinearModel.σ_ϵ, 
        β_0, i_max, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs[end]), 
        LinearModel.θ1S, LinearModel.θ2S, 
        LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
        "Linear Model: EnRML Posterior",
        "$(LinearModel.PLOTS_DIR)/enrml/gn_enrml_posterior.pdf";
        θs_t=LinearModel.θS_T,
        caption="Ensemble size: $N_e. "*L"\beta_0"*": $β_0."
    )

    Plotting.plot_lm_posterior_predictions(
        LinearModel.TS, ys[end], LinearModel.YS_T,
        LinearModel.TS_O, LinearModel.YS_O,
        "Linear Model: EnRML Posterior Predictions",
        "$(LinearModel.PLOTS_DIR)/enrml/gn_enrml_posterior_predictions.pdf"
    )

end

if lm_enrml

    const γ = 10

    θs, ys, Ss, λs = SimIntensiveInference.run_lm_enrml(
        LinearModel.f, LinearModel.g, π,  
        LinearModel.YS_O, LinearModel.σ_ϵ, 
        γ, i_max, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs[end]), 
        LinearModel.θ1S, LinearModel.θ2S, 
        LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
        "Linear Model: LM-EnRML Posterior",
        "$(LinearModel.PLOTS_DIR)/enrml/lm_enrml_posterior.pdf";
        θs_t=LinearModel.θS_T,
        caption="Ensemble size: $N_e. "*L"\gamma"*": $γ."
    )

    Plotting.plot_lm_posterior_predictions(
        LinearModel.TS, ys[end], LinearModel.YS_T,
        LinearModel.TS_O, LinearModel.YS_O,
        "Linear Model: LM-EnRML Posterior Predictions",
        "$(LinearModel.PLOTS_DIR)/enrml/lm_enrml_posterior_predictions.pdf"
    )

end