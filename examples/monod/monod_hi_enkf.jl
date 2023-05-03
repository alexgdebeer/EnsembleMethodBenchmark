"""Runs the EnKF on the MONOD model."""

include("monod_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior and ensemble size
const π = SimIntensiveInference.GaussianPrior(MONODModel.μ_π, MONODModel.Γ_π)
const N_e = 100

const MDA = false
const αs = [57.017, 35.0, 25.0, 20.0, 18.0, 15.0, 12.0, 8.0, 5.0, 3.0]

if MDA 

    θs = SimIntensiveInference.run_hi_enkf_mda(
        MONODModel.H, π, 
        MONODModel.XS, MONODModel.YS_O[:,:]', 
        MONODModel.σ_ϵ, αs, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        MONODModel.θ1S, MONODModel.θ2S, 
        MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
        "MONOD: Final HI-EnKF-MDA Posterior",
        "$(MONODModel.PLOTS_DIR)/enkf/hi_enkf_mda_posterior.pdf";
        caption="Ensemble size: $N_e."
    )

else

    θs = SimIntensiveInference.run_hi_enkf(
        MONODModel.H, π, 
        MONODModel.XS, MONODModel.YS_O[:,:]', 
        MONODModel.σ_ϵ, N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        MONODModel.θ1S, MONODModel.θ2S, 
        MONODModel.POST_MARG_θ1, MONODModel.POST_MARG_θ2,
        "MONOD: Final HI-EnKF Posterior",
        "$(MONODModel.PLOTS_DIR)/enkf/hi_enkf_posterior.pdf",
        caption="Ensemble size: $N_e."
    )

end