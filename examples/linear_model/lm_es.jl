"""Runs the ES on the linear model."""

include("linear_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(LinearModel.μ_π, LinearModel.Γ_π)
const N_e = 10_000

# Specify whether multiple data assimilation will occur, and if so, the α 
# values to use
const MDA = true
# const αs = [9.333, 7.0, 4.0, 2.0]
const αs = [57.017, 35.0, 25.0, 20.0, 18.0, 15.0, 12.0, 8.0, 5.0, 3.0]

if MDA

    θs = SimIntensiveInference.run_es_mda(
        LinearModel.f, LinearModel.g, π,  
        LinearModel.YS_O, 
        LinearModel.σ_ϵ, αs, N_e
    )

    ys = reduce(vcat, [LinearModel.f(θ) for θ ∈ eachcol(θs)])

    Plotting.plot_lm_state_evolution(
        ys, 
        LinearModel.TS, LinearModel.YS_T, 
        LinearModel.TS_O, LinearModel.YS_O, 
        "Linear Model: ES-MDA Posterior Predictions",
        "$(LinearModel.PLOTS_DIR)/es/es_mda_posterior_predictions.pdf"
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        LinearModel.θ1S, LinearModel.θ2S, 
        LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
        "Linear Model: ES-MDA Posterior",
        "$(LinearModel.PLOTS_DIR)/es/es_mda_posterior.pdf";
        θs_t=LinearModel.θS_T,
        caption="Ensemble size: $N_e."
    )

else

    θs = SimIntensiveInference.run_es(
        LinearModel.f, LinearModel.g, π,  
        LinearModel.YS_O, 
        LinearModel.σ_ϵ, N_e
    )

    ys = reduce(vcat, [LinearModel.f(θ) for θ ∈ eachcol(θs)])

    Plotting.plot_lm_state_evolution(
        ys, 
        LinearModel.TS, LinearModel.YS_T, 
        LinearModel.TS_O, LinearModel.YS_O, 
        "Linear Model: ES Posterior Predictions",
        "$(LinearModel.PLOTS_DIR)/es/es_posterior_predictions.pdf"
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        LinearModel.θ1S, LinearModel.θ2S, 
        LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
        "Linear Model: ES Posterior",
        "$(LinearModel.PLOTS_DIR)/es/es_posterior.pdf";
        θs_t=LinearModel.θS_T,
        caption="Ensemble size: $N_e."
    )

end