"""Runs the EnKF on the LV model."""

using SimIntensiveInference

include("model_setup.jl")
include("../../plotting.jl")

N_e = 100

θs, us = SimIntensiveInference.run_enkf(
    LVModel.f, LVModel.b, π, LVModel.Y_0, 
    copy(LVModel.TS_O), LVModel.YS_O, 
    LVModel.σ_ϵ, N_e
)

Plotting.plot_approx_posterior(
    eachcol(θs), 
    LVModel.AS, LVModel.BS, 
    LVModel.POST_MARG_A, LVModel.POST_MARG_B,
    "LV: Final EnKF Posterior",
    "$(LVModel.PLOTS_DIR)/enkf/enkf_posterior.pdf";
    θs_t=LVModel.θS_T,
    caption="Ensemble size: $N_e."
)

Plotting.plot_lv_state_evolution(
    us, LVModel.TS, LVModel.YS_T, LVModel.TS_O, LVModel.YS_O, 
    "LV: EnKF State Evolution",
    "$(LVModel.PLOTS_DIR)/enkf/enkf_state_evolution.pdf"
)