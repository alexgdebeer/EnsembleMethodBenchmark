using Distributions
using LinearAlgebra
using Statistics

import PyPlot
import Seaborn

include("lokta_volterra_model.jl")
include("sim_intensive_inference/sim_intensive_inference.jl")

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")


function run_enkf(
    f::Function,
    g::Function,
    π_u::SimIntensiveInference.AbstractPrior,
    ts::Vector,
    t_1::Real,
    ys_list::Vector,
    σ_ϵ::Real,
    N_e::Int
)

    # Generate an initial sample of states from the prior
    us_e = SimIntensiveInference.sample(π_u, n=N_e)

    # Define a vector that offsets the times by 1
    ts_0 = [0.0, ts[1:(end-1)]...]

    us_e_comb = []

    for (i, (t_0, t_1, y)) ∈ enumerate(zip(ts_0, ts, ys_list))

        # Run each ensemble member forward in time 
        us_e_cur = [f(θs; x_0=u, t_0=t_0, t_1=t_1) for u ∈ us_e]
        us_e_cur = [reshape(u, (div(length(u), 2), 2)) for u ∈ us_e_cur]
        
        # Save the combined state vectors for each ensemble member
        if i == 1
            us_e_comb = copy(us_e_cur)
        else
            us_e_comb = [[u_p; u[2:end,:]] for (u_p, u) ∈ zip(us_e_comb, us_e_cur)]
        end

        # Extract the forecast states and generate the predictions 
        us_ef = [u[end,:] for u ∈ us_e_cur]
        ys_ef = [g(θs, u) for u ∈ us_ef]

        # Generate a set of perturbed data vectors 
        Γ_ϵϵ = σ_ϵ^2 * Matrix(I, length(y), length(y))
        ys_p = [rand(MvNormal(y, Γ_ϵϵ)) for _ ∈ 1:N_e]

        # Compute the covariance of the predicted data 
        U_c = hcat(us_ef...) * (I - ones(N_e, N_e)/N_e)
        Y_c = hcat(ys_ef...) * (I - ones(N_e, N_e)/N_e)
        Γ_uy_e = 1/(N_e-1)*U_c*Y_c'
        Γ_yy_e = 1/(N_e-1)*Y_c*Y_c'

        K = Γ_uy_e * inv(Γ_yy_e + Γ_ϵϵ)
        us_e = [u + K*(y_p-y_e) for (u, y_e, y_p) ∈ zip(us_ef, ys_ef, ys_p)]

    end

    # Run each model to the end if necessary
    if ts[end] < t_1
        us_e_cur = [f(θs; x_0=u, t_0=ts[end], t_1=t_1) for u ∈ us_e]
        us_e_cur = [reshape(u, (div(length(u), 2), 2)) for u ∈ us_e_cur]
        us_e_comb = [[u_p; u[2:end,:]] for (u_p, u) ∈ zip(us_e_comb, us_e_cur)]
    end

    return us_e_comb

end

# Define the values of the parameters 
const θs = [1.0, 1.0]
const σ_ϵ = 0.5

ts, ys_true, is_obs, ts_obs, ys_obs = LVModel.generate_data(θs, σ_ϵ)

# Format the observations
ts_obs = ts_obs[1:LVModel.N_DATA]
ys_obs = [[ys_obs[i], ys_obs[i+LVModel.N_DATA]] for i ∈ 1:LVModel.N_DATA]

# Define parameters of the prior for the model state 
const μ_u = [0.75, 0.75]
const σ_u = 0.5
const Σ_u = σ_u^2 .* Matrix(1.0I, 2, 2)
const π_u = SimIntensiveInference.GaussianPrior(μ_u, Σ_u)

# Define ensemble size 
const N_e = 100

us_e_comb = run_enkf(LVModel.f, LVModel.g, π_u, ts_obs, LVModel.T_1, ys_obs, σ_ϵ, N_e)

fig, ax = PyPlot.subplots(1, 2)

xs = hcat([u[:, 1] for u ∈ us_e_comb]...)
ys = hcat([u[:, 2] for u ∈ us_e_comb]...)

x_qs = hcat([quantile(x, [0.05, 0.95]) for x ∈ eachrow(xs)]...)'
y_qs = hcat([quantile(y, [0.05, 0.95]) for y ∈ eachrow(ys)]...)'

for u ∈ us_e_comb
    ax[1].plot(ts, u[:, 1], color="gray", alpha=0.5, zorder=2)
    ax[2].plot(ts, u[:, 2], color="gray", alpha=0.5, zorder=2)
end

ax[1].plot(ts, ys_true[1:151], color="k", ls="--", zorder=3)
ax[2].plot(ts, ys_true[152:end], color="k", ls="--", zorder=3)

ax[1].scatter(ts_obs, [y[1] for y ∈ ys_obs], color="k", marker="x", zorder=3)
ax[2].scatter(ts_obs, [y[2] for y ∈ ys_obs], color="k", marker="x", zorder=3)

ax[1].plot(ts, x_qs[:, 1], color="red", zorder=3)
ax[1].plot(ts, x_qs[:, 2], color="red", zorder=3)
ax[2].plot(ts, y_qs[:, 1], color="red", zorder=3)
ax[2].plot(ts, y_qs[:, 2], color="red", zorder=3)

ax[1].set_ylim((-1, 4))
ax[2].set_ylim((-1, 4))

PyPlot.savefig("test.pdf")