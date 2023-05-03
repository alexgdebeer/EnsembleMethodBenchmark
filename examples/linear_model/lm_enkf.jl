"""Runs the EnKF for parameter estimation on the linear model."""

using Distributions
using LinearAlgebra

include("linear_model.jl")
include("../../plotting.jl")

const ΔT = LinearModel.ΔT

# Define the model function 
const f(u, θ, t_0, t_1) = u + θ * (t_1-t_0)

# Define the times at which observations are made, and the corresponding 
# observations
const ts = [0.0, LinearModel.TS_O...]
const ys = LinearModel.YS_O

# Define the prior and ensemble size
const σ_ϵ = LinearModel.σ_ϵ
const N_e = 10_000

const n_us = 1

function kf()

    # Generate an initial ensemble from the prior
    us_e = rand(Normal(0.0, 3.0), N_e)
    θs_e = rand(Normal(0.0, 3.0), N_e)

    uθs_e = 0

    for (i, (t_0, t_1, y)) ∈ enumerate(zip(ts[1:(end-1)], ts[2:end], ys))

        # Run each ensemble member forward in time and extract the final state 
        us_e = [f(u, θ, t_0, t_1) for (u, θ) ∈ zip(us_e, θs_e)]

        ys_e = copy(us_e)

        # Generate a set of perturbed data vectors 
        ys_p = rand(Normal(y, σ_ϵ), N_e)

        # Compute the Kalman gain
        U_c = vcat(us_e', θs_e') * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Y_c = ys_e' * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Γ_uy_e = 1/(N_e-1)*U_c*Y_c'
        Γ_y_e = 1/(N_e-1)*Y_c*Y_c'
        K = Γ_uy_e * (Γ_y_e + σ_ϵ^2)^-1
        
        # Update the ensemble
        uθs_e = vcat(us_e', θs_e') + K*(ys_p'-ys_e')
        us_e, θs_e = uθs_e[1, :], uθs_e[2, :]

        @info("Iteration $i complete.")

    end

    # Find the intercept that each final state and slope correspond to
    uθs_e[1,:] .-= uθs_e[2,:].*(LinearModel.TS_O[end])

    return uθs_e

end

θs = kf()

Plotting.plot_approx_posterior(
    eachcol(θs), 
    LinearModel.θ1S, LinearModel.θ2S, 
    LinearModel.POST_MARG_θ1, LinearModel.POST_MARG_θ2,
    "Linear Model: Final EnKF Posterior",
    "$(LinearModel.PLOTS_DIR)/enkf/enkf_posterior.pdf";
    θs_t=LinearModel.θS_T,
    caption="Ensemble size: $N_e."
)