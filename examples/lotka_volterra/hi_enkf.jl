"""Runs the half-iteration EnKF algorithm on the LV model."""

using Revise

include("lv_model.jl")
include("../../plotting.jl")
include("../../sim_intensive_inference/sim_intensive_inference.jl")

Revise.track("examples/lotka_volterra/lv_model.jl")
Revise.track("plotting.jl")
Revise.track("sim_intensive_inference/sim_intensive_inference.jl")

import .LVModel
import .Plotting
import .SimIntensiveInference

# Define the prior, ensemble size and number of states
N_e = 100
N_u = 2

θs, us = SimIntensiveInference.run_hi_enkf(
    LVModel.f, LVModel.b, LVModel.π,
    LVModel.TS_O, LVModel.YS_O, 
    LVModel.σ_ϵ, N_e, N_u
)

Plotting.plot_approx_posterior(
    eachcol(θs[:,:,end]), 
    LVModel.AS, LVModel.BS, 
    LVModel.POST_MARG_A, LVModel.POST_MARG_B, 
    "LV: Final HI-EnKF Posterior",
    "$(LVModel.PLOTS_DIR)/enkf/hi_enkf_posterior.pdf";
    θs_t=LVModel.θS_T,
    caption="Ensemble size: $N_e."
)