import DelimitedFiles
import DifferentialEquations
import LinearAlgebra
import Optim
import PyPlot
import Statistics

include("sim_intensive_inference.jl")


PyPlot.rc("text", usetex = true)
PyPlot.rc("font", family = "serif")


# Define start year, end year and timespan of interest
t_start = 1951
t_end = 2014
ts = range(t_start, t_end, 631)

# Define constants for unit conversions
SECS_PER_DAY = 60.0 * 60.0 * 24.0
SECS_PER_YEAR = SECS_PER_DAY * 365.25

# Define atmospheric pressure, gravitational acceleration, density of water
P_atm = 101_325.0
g = -9.81
ρ = 997.0

FNAME_WATER_LEVELS = joinpath("data", "water_levels.txt")
FNAME_TEMPERATURES = joinpath("data", "temperatures.txt")
FNAME_Q_TOTAL = joinpath("data", "q_total.txt")
FNAME_Q_RHYOLITE = joinpath("data", "q_rhyolite.txt")


"""Carries out linear interpolation for a single data point."""
function interpolate(
    x::AbstractVector{<:Real}, 
    y::AbstractVector{<:Real}, 
    x_i::Real
)::Float64

    if (x_i < x[1]) || (x_i > x[end])
        error("Point to interpolate at lies outside range of data.")
    end

    for (x_1, x_2, y_1, y_2) ∈ zip(x[1:end-1], x[2:end], y[1:end-1], y[2:end])
        if (x_1 <= x_i <= x_2)
            return y_1 + (x_i - x_1) * (y_2 - y_1) / (x_2 - x_1)
        end
    end

end


"""Carries out linear interpolation for a vector of data points."""
function interpolate(
    x::AbstractVector{<:Real}, 
    y::AbstractVector{<:Real}, 
    x_is::AbstractVector{<:Real}
)::AbstractVector{Float64}

    return [interpolate(x, y, x_i) for x_i in x_is]

end


"""Reads in and interpolates data as required."""
function read_data()

    function read_file(fname::AbstractString)::AbstractMatrix
        return DelimitedFiles.readdlm(fname, ',', Float64, '\n', skipstart = 1)
    end

    # Read in water levels, temperatures and extraction rates
    water_levels = read_file(FNAME_WATER_LEVELS)
    temperatures = read_file(FNAME_TEMPERATURES)
    qs_total = read_file(FNAME_Q_TOTAL)
    qs_rhyolite = read_file(FNAME_Q_RHYOLITE)

    # Convert water level to hydrostatic pressure in the middle of the reservoir
    ts_P_obs = water_levels[:, 1]
    P_obs = P_atm .- ρ .* g .* water_levels[:, 2] ./ 2
    
    ts_T_obs = temperatures[:, 1]
    T_obs = temperatures[:, 2]

    ts_q_tot = qs_total[:, 1]
    qs_tot = qs_total[:, 2]
    ts_q_rhy = qs_rhyolite[:, 1]
    qs_rhy = qs_rhyolite[:, 2]

    # Calculate the non-rhyolite extraction using interpolation, and convert
    # from units of tonnes/day to kg/s
    qs = interpolate(ts_q_tot, qs_tot, ts) .- interpolate(ts_q_rhy, qs_rhy, ts)
    qs .*= (1000.0 / SECS_PER_DAY)

    # Calculate rate of change of extraction over time using finite differences
    dqdts = zeros(Float64, length(qs))
    dqdts[1] = (qs[2] - qs[1]) / (ts[2] .- ts[1])
    dqdts[2:end-1] = (qs[3:end] .- qs[1:end-2]) ./ (ts[3:end] .- ts[1:end-2])
    dqdts[end] = (qs[end] - qs[end-1]) / (ts[end] - ts[end-1])

    # Convert rate of change of extraction to units of kg/s^2
    dqdts ./= SECS_PER_YEAR

    return ts_P_obs, P_obs, ts_T_obs, T_obs, ts, qs, dqdts

end


"""Generates the observation operator."""
function gen_obs_operator(
    ts::AbstractVector, 
    ts_P_obs::AbstractVector, 
    ts_T_obs::AbstractVector
)::Matrix

    # Define upper and lower blocks of zeros
    G_u = zeros(length(ts_P_obs), length(ts))
    G_l = zeros(length(ts_T_obs), length(ts))

    # Define blocks for pressures and temperatures
    G_p = zeros(length(ts_P_obs), length(ts))
    G_t = zeros(length(ts_T_obs), length(ts))
    
    for (i, t) in enumerate(ts_P_obs)
        G_p[i, argmin(abs.(ts .- t))] = 1
    end

    for (i, t) in enumerate(ts_T_obs)
        G_t[i, argmin(abs.(ts .- t))] = 1
    end

    return [G_p G_u; G_l G_t]

end


"""Solves model for a given set of parameters, and returns the modelled 
pressures and temperatures."""
function f(
    θ::Union{AbstractArray, AbstractVector}
)::Vector

    """Returns pressure derivative at a given time."""
    function dpdt(P::Real, params::Vector{<:Real}, t::Real)::Real

        # Unpack parameters
        a_p, b_p, c_p, P_0 = params

        # Find rate of extraction at current time
        q = interpolate(ts, qs, t)
        dqdt = interpolate(ts, dqdts, t)

        # Return pressure derivative
        return -a_p*q - b_p*(P - P_0) - c_p*dqdt

    end

    """Returns temperature derivative at a given time."""
    function dtdt(T::Real, params::Vector{<:Real}, t::Real)::Real

        # Unpack parameters
        a_p, b_p, P_0, a_t, b_t, T_0 = params 

        # Find current pressure
        P = interpolate(ts, Ps, t)
        
        # Define reference temperature (assuming that the temperature of the 
        # cold water outside the reservoir is 30ºC)
        T_ref = P > P_0 ? T : 30.0

        # Return temperature derivative
        return -a_t*(b_p / a_p)*(P - P_0)*(T_ref - T) - b_t*(T - T_0)

    end

    # Rearrange parameters
    a_p, b_p, c_p, P_0, a_t, b_t, T_0 = θ
    p_params = [a_p, b_p, c_p, P_0]
    t_params = [a_p, b_p, P_0, a_t, b_t, T_0]

    tspan = (t_start, t_end)

    # Formulate and solve pressure ODE
    P_prob = DifferentialEquations.ODEProblem(dpdt, P_0, tspan, p_params)

    P_sol = DifferentialEquations.solve(
        P_prob, 
        DifferentialEquations.RK4(); 
        tstops = ts, saveat = ts, maxiters = 100_000
    )
    
    Ps = P_sol.u

    # Formulate and solve temperature ODE
    T_prob = DifferentialEquations.ODEProblem(dtdt, T_0, tspan, t_params)

    T_sol = DifferentialEquations.solve(
        T_prob, 
        DifferentialEquations.RK4(); 
        tstops = ts, saveat = ts
    )

    Ts = T_sol.u

    return vcat(Ps, Ts)

end


"""Calculates the normalised sum of squared differences between a set of 
observations and modelled values."""
function standardised_sum_sq(
    y_m::AbstractVector,
    y_obs::AbstractVector
)

    y_m = [y_m[1:n_P_obs], y_m[(n_P_obs+1):end]]
    y_obs = [y_obs[1:n_P_obs], y_obs[(n_P_obs+1):end]]

    # Calculate the means and standard deviations of each set of data
    μs = [Statistics.mean(y) for y in y_obs]
    σs = [Statistics.std(y) for y in y_obs]

    # Normalise the observations and modelled values
    y_m = [((y .- μ) ./ σ) for (y, μ, σ) ∈ zip(y_m, μs, σs)]
    y_obs = [((y .- μ) ./ σ) for (y, μ, σ) ∈ zip(y_obs, μs, σs)]

    # Return the sum of the sum of squared differences between each set of data
    # and the model
    return sum(sum((y_d .- y_m).^2) for (y_d, y_m) ∈ zip(y_obs, y_m))

end


"""Function to optimise."""
function func_to_optimise(
    θs::AbstractVector
)::Float64

    # Return the modelled temperatures and pressures at the times of interest
    y_m = G * f(θs)

    return standardised_sum_sq(y_m, vcat(P_obs, T_obs))

end

ts_P_obs, P_obs, ts_T_obs, T_obs, ts, qs, dqdts = read_data()

n_P_obs = length(P_obs)
n_T_obs = length(T_obs)
n_obs = n_P_obs + n_T_obs

# Generate observation operator
G = gen_obs_operator(ts, ts_P_obs, ts_T_obs)

x₀ = [13.0, 0.1, 1.7e9, 1.54e6, 1.6e-4, 0.8e-2, 150.0]

# Optimise 
sol = Optim.optimize(func_to_optimise, x₀, Optim.NelderMead())

θs_cal = Optim.minimizer(sol)

ys_cal = f(θs_cal)

Ps_cal, Ts_cal = ys_cal[1:length(ts)], ys_cal[(length(ts)+1):end]

fig, ax = PyPlot.subplots(1, 2)
fig.suptitle("Calibrated LPM", fontsize = 20)
fig.set_size_inches(10, 4)

ax[1].set_title("Pressures")
ax[2].set_title("Temperatures")

ax[1].plot(ts, Ps_cal ./ 1e6, color = "tab:blue", zorder = 1, label = "Calibrated model")
ax[1].scatter(ts_P_obs, P_obs ./ 1e6, marker = "x", color = "k", zorder = 2, label = "Observations")

ax[2].plot(ts, Ts_cal, color = "tab:blue", zorder = 1, label = "Calibrated model")
ax[2].scatter(ts_T_obs, T_obs, marker = "x", color = "k", zorder = 2, label = "Observations")

ax[1].set_xlabel("\$t\$ (years)")
ax[2].set_xlabel("\$t\$ (years)")

ax[1].set_ylabel("\$P\$ (MPa)")
ax[2].set_ylabel("\$T\$ (°C)")

ax[1].legend()
ax[2].legend()

PyPlot.savefig("plots/calibrated_lpm.pdf")

# # Define mean and covariance of prior
# μ = [13.0, 0.1, 1.7e9, 1.54e6, 1.6e-4, 0.8e-2, 150.0]
# σ = 0.1 .* μ
# σ[4] = 0.01e+6
# Σ = LinearAlgebra.diagm(σ.^2)

# prior = SimIntensiveInference.MVNormalPrior(μ, Σ)

# # Define a noise model
# μ = zeros(n_obs)
# σ = vcat(repeat([1500], n_P_obs), repeat([1.5], n_T_obs))
# Σ = LinearAlgebra.diagm(σ.^2)

# noise = SimIntensiveInference.NormalNoise(μ, Σ)

# n = 1000
# α = 0.05

# θs, ys, ds, inds = SimIntensiveInference.run_abc(
#     prior, 
#     forward_model, 
#     noise,
#     vcat(P_obs, T_obs), 
#     G,
#     SimIntensiveInference.standardised_sum_sq,
#     n, 
#     α;
#     divs = [n_P_obs + 1]
# )

# ys_filtered = ys[inds, :]
# ys_rejected = ys[setdiff(1:end, inds), :]

# # Plot the results
# for i ∈ 1:(n - length(inds))
#     PyPlot.plot(ts, ys_rejected[i, 1:length(ts)], color = "tab:gray", zorder = 1)
# end

# for i ∈ 1:length(inds)
#     PyPlot.plot(ts, ys_filtered[i, 1:length(ts)], color = "tab:green", zorder = 2)
# end

# PyPlot.scatter(ts_P_obs, P_obs, c = "k", zorder = 3)

# PyPlot.savefig("abc_pressures.pdf")

# """Calculates the normalised sum of squared differences between a set of 
# observations and modelled values."""
# function standardised_sum_sq(
#     y_model::Vector,
#     y_obs::Vector;
#     divs::Vector = []
# )

#     divs = vcat(1, divs, length(y_obs)+1)

#     y_model = [y_model[s:(e-1)] for (s, e) ∈ zip(divs[1:(end-1)], divs[2:end])]
#     y_obs = [y_obs[s:(e-1)] for (s, e) ∈ zip(divs[1:(end-1)], divs[2:end])]

#     # Calculate the means and standard deviations of each set of data
#     μs = [Statistics.mean(y) for y in y_obs]
#     σs = [Statistics.std(y) for y in y_obs]

#     # Normalise the observations and modelled values
#     y_model = [((y .- μ) ./ σ) for (y, μ, σ) ∈ zip(y_model, μs, σs)]
#     y_obs = [((y .- μ) ./ σ) for (y, μ, σ) ∈ zip(y_obs, μs, σs)]

#     # Return the sum of the sum of squared differences between each set of data
#     # and the model
#     return sum(sum((y_d .- y_m).^2) for (y_d, y_m) ∈ zip(y_obs, y_model))

# end