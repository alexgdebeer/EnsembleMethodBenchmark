"""Runs the ES on the MONOD model."""

include("monod_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(MONODModel.μ_π, MONODModel.Γ_π)
const N_e = 10_000

# Specify whether multiple data assimilation will occur, and if so, the α 
# values to use
const MDA = false
#const αs = [9.333, 7.0, 4.0, 2.0]
const αs = [57.017, 35.0, 25.0, 20.0, 18.0, 15.0, 12.0, 8.0, 5.0, 3.0]

if MDA

    θs = SimIntensiveInference.run_ensemble_smoother_mda(
        MONODModel.f, 
        MONODModel.g,
        π,  
        MONODModel.YS_O, 
        MONODModel.σ_ϵ, 
        αs,
        N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        MONODModel.θ1S, MONODModel.θ2S, 
        MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
        "MONOD Model: ES MDA Posterior",
        "$(MONODModel.PLOTS_DIR)/es/es_mda_posterior.pdf";
        caption="Ensemble size: $N_e."
    )

else

    θs = SimIntensiveInference.run_ensemble_smoother(
        MONODModel.f, 
        MONODModel.g,
        π,  
        MONODModel.YS_O, 
        MONODModel.σ_ϵ, 
        N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        MONODModel.θ1S, MONODModel.θ2S, 
        MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
        "MONOD Model: ES Posterior",
        "$(MONODModel.PLOTS_DIR)/es/es_posterior.pdf";
        caption="Ensemble size: $N_e."
    )

end