"""Runs EnRML on the LV model."""

using LaTeXStrings

include("lv_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(LVModel.μ_π, LVModel.Γ_π)

const batch_enrml = true
const lm_enrml = true

const i_max = 16
const N_e = 100

if batch_enrml

    const β_0 = 0.5

    θs, ys, Ss, βs = SimIntensiveInference.run_batch_enrml(
        LVModel.f, LVModel.g, π,  
        reduce(vcat, LVModel.YS_O), LVModel.σ_ϵ, 
        β_0, i_max, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs[end]), 
        LVModel.AS, LVModel.BS, 
        LVModel.POST_MARG_A, LVModel.POST_MARG_B,
        "LV: GN-EnRML Posterior",
        "$(LVModel.PLOTS_DIR)/enrml/gn_enrml_posterior.pdf";
        θs_t=LVModel.θS_T,
        caption="Ensemble size: $N_e. "*L"\beta_0"*": $β_0."
    )

    Plotting.plot_lv_posterior_predictions(
        LVModel.TS, ys[end], LVModel.YS_T, 
        LVModel.TS_O, LVModel.YS_O, 
        "LV: GN-EnRML Posterior Predictions", 
        "$(LVModel.PLOTS_DIR)/enrml/gn_enrml_posterior_predictions.pdf"
    )

end

if lm_enrml

    const γ = 10

    θs, ys, Ss, λs = SimIntensiveInference.run_lm_enrml(
        LVModel.f, LVModel.g, π, 
        reduce(vcat, LVModel.YS_O), LVModel.σ_ϵ, 
        γ, i_max, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs[end]), 
        LVModel.AS, LVModel.BS, 
        LVModel.POST_MARG_A, LVModel.POST_MARG_B,
        "LV: LM-EnRML Posterior",
        "$(LVModel.PLOTS_DIR)/enrml/lm_enrml_posterior.pdf";
        θs_t=LVModel.θS_T,
        caption="Ensemble size: $N_e. "*L"\gamma:"*" $γ."
    )

    Plotting.plot_lv_posterior_predictions(
        LVModel.TS, ys[end], LVModel.YS_T, 
        LVModel.TS_O, LVModel.YS_O, 
        "LV: LM-EnRML Posterior Predictions", 
        "$(LVModel.PLOTS_DIR)/enrml/lm_enrml_posterior_predictions.pdf"
    )

end