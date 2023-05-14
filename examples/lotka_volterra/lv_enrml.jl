"""Runs EnRML on the LV model."""

include("lv_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(LVModel.μ_π, LVModel.Γ_π)

const batch_enrml = false
const lm_enrml = true

if batch_enrml

    const β_0 = 0.5
    const N_e = 100

    θs, ys = SimIntensiveInference.run_batch_enrml(
        LVModel.f, LVModel.g, π,  
        reduce(vcat, LVModel.YS_O), LVModel.σ_ϵ, 
        β_0, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        LVModel.AS, LVModel.BS, 
        LVModel.POST_MARG_A, LVModel.POST_MARG_B,
        "LV: EnRML Posterior",
        "$(LVModel.PLOTS_DIR)/enrml/enrml_posterior.pdf";
        θs_t=LVModel.θS_T,
        caption="Ensemble size: $N_e."
    )

    Plotting.plot_lv_posterior_predictions(
        LVModel.TS, ys, LVModel.YS_T, 
        LVModel.TS_O, LVModel.YS_O, 
        "LV: EmRML Posterior Predictions", 
        "$(LVModel.PLOTS_DIR)/enrml/enrml_posterior_predictions.pdf"
    )

end

if lm_enrml

    const γ = 10
    const l_max = 16
    const N_e = 100

    θs, ys = SimIntensiveInference.run_lm_enrml(
        LVModel.f, LVModel.g, π, 
        reduce(vcat, LVModel.YS_O), LVModel.σ_ϵ, 
        γ, l_max, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs[end]), 
        LVModel.AS, LVModel.BS, 
        LVModel.POST_MARG_A, LVModel.POST_MARG_B,
        "LV: LM-EnRML Posterior",
        "$(LVModel.PLOTS_DIR)/enrml/lm_enrml_posterior.pdf";
        θs_t=LVModel.θS_T,
        caption="Ensemble size: $N_e."
    )

    Plotting.plot_lv_posterior_predictions(
        LVModel.TS, ys[end], LVModel.YS_T, 
        LVModel.TS_O, LVModel.YS_O, 
        "LV: LM-EmRML Posterior Predictions", 
        "$(LVModel.PLOTS_DIR)/enrml/lm_enrml_posterior_predictions.pdf"
    )

end