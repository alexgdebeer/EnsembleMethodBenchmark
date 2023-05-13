"""Runs the ES on the Lotka-Volterra model."""

include("lv_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(LVModel.μ_π, LVModel.Γ_π)
const N_e = 100

# Specify whether multiple data assimilation will occur, and if so, the α 
# values to use
const MDA = true
# const αs = [9.333, 7.0, 4.0, 2.0]
const αs = [16.0 for _ ∈ 1:16]

if MDA

    θs, ys = SimIntensiveInference.run_es_mda(
        LVModel.f, LVModel.g, π,  
        reduce(vcat, LVModel.YS_O), LVModel.σ_ϵ, 
        αs, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs[end]), 
        LVModel.AS, LVModel.BS, 
        LVModel.POST_MARG_A, LVModel.POST_MARG_B,
        "LV: ES-MDA Posterior",
        "$(LVModel.PLOTS_DIR)/es/es_mda_posterior.pdf";
        θs_t=LVModel.θS_T,
        caption="Ensemble size: $N_e. Iterations: $(length(αs))."
    )

    Plotting.plot_lv_posterior_predictions(
        LVModel.TS, ys[end], LVModel.YS_T, LVModel.TS_O, LVModel.YS_O, 
        "LV: ES-MDA Posterior Predictions", 
        "$(LVModel.PLOTS_DIR)/es/es_mda_posterior_predictions.pdf"
    )

else

    θs = SimIntensiveInference.run_es(
        LVModel.f, LVModel.g, π,  
        reduce(vcat, LVModel.YS_O), 
        LVModel.σ_ϵ, N_e
    )

    ys = reduce(vcat, [LVModel.f(θ) for θ ∈ eachcol(θs)])

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        LVModel.AS, LVModel.BS, 
        LVModel.POST_MARG_A, LVModel.POST_MARG_B,
        "LV: ES Posterior",
        "$(LVModel.PLOTS_DIR)/es/es_posterior.pdf";
        θs_t=LVModel.θS_T,
        caption="Ensemble size: $N_e."
    )

    Plotting.plot_lv_posterior_predictions(
        LVModel.TS, ys, LVModel.YS_T, LVModel.TS_O, LVModel.YS_O, 
        "LV: ES Posterior Predictions", 
        "$(LVModel.PLOTS_DIR)/es/es_posterior_predictions.pdf"
    )

end