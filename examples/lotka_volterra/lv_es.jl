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
#const αs = [9.333, 7.0, 4.0, 2.0]
const αs = [57.017, 35.0, 25.0, 20.0, 18.0, 15.0, 12.0, 8.0, 5.0, 3.0]

if MDA

    θs = SimIntensiveInference.run_es_mda(
        LVModel.f, LVModel.g, π,  
        reduce(vcat, LVModel.YS_O), 
        LVModel.σ_ϵ, αs, N_e
    )

    ys = reduce(vcat, [LVModel.f(θ) for θ ∈ eachcol(θs)])

    Plotting.plot_lv_state_evolution(
        ys, LVModel.TS, LVModel.YS_T, LVModel.TS_O, LVModel.YS_O, 
        "LV: ES-MDA Posterior Predictions", 
        "$(LVModel.PLOTS_DIR)/es/es_mda_posterior_predictions.pdf"
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        LVModel.AS, LVModel.BS, 
        LVModel.POST_MARG_A, LVModel.POST_MARG_B,
        "LV: ES-MDA Posterior",
        "$(LVModel.PLOTS_DIR)/es/es_mda_posterior.pdf";
        θs_t=LVModel.θS_T,
        caption="Ensemble size: $N_e."
    )

else

    θs = SimIntensiveInference.run_es(
        LVModel.f, LVModel.g, π,  
        reduce(vcat, LVModel.YS_O), 
        LVModel.σ_ϵ, N_e
    )

    ys = reduce(vcat, [LVModel.f(θ) for θ ∈ eachcol(θs)])

    Plotting.plot_lv_state_evolution(
        ys, LVModel.TS, LVModel.YS_T, LVModel.TS_O, LVModel.YS_O, 
        "LV: ES Posterior Predictions", 
        "$(LVModel.PLOTS_DIR)/es/es_posterior_predictions.pdf"
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        LVModel.AS, LVModel.BS, 
        LVModel.POST_MARG_A, LVModel.POST_MARG_B,
        "LV Model: ES Posterior",
        "$(LVModel.PLOTS_DIR)/es/es_posterior.pdf";
        θs_t=LVModel.θS_T,
        caption="Ensemble size: $N_e."
    )

end