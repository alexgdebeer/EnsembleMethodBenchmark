"""Runs EnRML on the MONOD model."""

using LaTeXStrings

include("monod_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(MONODModel.μ_π, MONODModel.Γ_π)

const batch_enrml = true
const lm_enrml = true

const i_max = 16
const N_e = 100

if batch_enrml

    const β_0 = 0.5

    θs, ys, Ss, βs = SimIntensiveInference.run_batch_enrml(
        MONODModel.f, MONODModel.g, π, 
        MONODModel.YS_O, MONODModel.σ_ϵ,
        β_0, i_max, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs[end]), 
        MONODModel.θ1S, MONODModel.θ2S, 
        MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
        "MONOD: GN-EnRML Posterior",
        "$(MONODModel.PLOTS_DIR)/enrml/gn_enrml_posterior.pdf";
        caption="Ensemble size: $N_e. "*L"\beta_0:"*" $β_0."
    )

    Plotting.plot_monod_posterior_predictions(
        MONODModel.XS, ys[end], 
        MONODModel.XS_O, MONODModel.YS_O, 
        "MONOD: GN-EnRML Posterior Predictions",
        "$(MONODModel.PLOTS_DIR)/enrml/gn_enrml_posterior_predictions.pdf"
    )

end

if lm_enrml

    const γ = 10

    θs, ys, Ss, βs = SimIntensiveInference.run_lm_enrml(
        MONODModel.f, MONODModel.g, π, 
        MONODModel.YS_O, MONODModel.σ_ϵ,
        γ, i_max, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs[end]), 
        MONODModel.θ1S, MONODModel.θ2S, 
        MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
        "MONOD: LM-EnRML Posterior",
        "$(MONODModel.PLOTS_DIR)/enrml/lm_enrml_posterior.pdf";
        caption="Ensemble size: $N_e. "*L"\gamma:"*" $γ."
    )

    Plotting.plot_monod_posterior_predictions(
        MONODModel.XS, ys[end], 
        MONODModel.XS_O, MONODModel.YS_O, 
        "MONOD: LM-EnRML Posterior Predictions",
        "$(MONODModel.PLOTS_DIR)/enrml/lm_enrml_posterior_predictions.pdf"
    )

end