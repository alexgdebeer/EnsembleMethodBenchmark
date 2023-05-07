"""Runs the half-iteration EnKF algorithm on the linear model."""

include("linear_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

# Define the prior, ensemble size and number of states
const π = SimIntensiveInference.GaussianPrior(LinearModel.μ_π, LinearModel.Γ_π)
const N_e = 10_000
const N_u = 1

const MDA = false
const αs = [9.333, 7.0, 4.0, 2.0]

if MDA 

    θs = SimIntensiveInference.run_hi_enkf_mda(
        LinearModel.H, π, 
        LinearModel.TS_O, LinearModel.YS_O[:,:]', 
        LinearModel.σ_ϵ, αs,N_e
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        LinearModel.θ1S, LinearModel.θ2S, 
        LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
        "Linear Model: Final HI-EnKF-MDA Posterior",
        "$(LinearModel.PLOTS_DIR)/enkf/hi_enkf_mda_posterior.pdf";
        θs_t=LinearModel.θS_T,
        caption="Ensemble size: $N_e."
    )

else

    θs, us = SimIntensiveInference.run_hi_enkf(
        LinearModel.a, LinearModel.b, π, 
        LinearModel.TS_O, LinearModel.YS_O[:,:]', 
        LinearModel.σ_ϵ, N_e, N_u
    )

    Plotting.plot_approx_posterior(
        eachcol(θs), 
        LinearModel.θ1S, LinearModel.θ2S, 
        LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
        "Linear Model: Final HI-EnKF Posterior",
        "$(LinearModel.PLOTS_DIR)/enkf/hi_enkf_posterior.pdf";
        θs_t=LinearModel.θS_T,
        caption="Ensemble size: $N_e."
    )

end