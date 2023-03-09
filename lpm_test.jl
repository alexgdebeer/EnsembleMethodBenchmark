import DelimitedFiles
import DifferentialEquations
import Optim
import PyPlot

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


"""Solves model for a given set of parameters, and returns the modelled 
pressures and temperatures."""
function solve_model(
    params::Vector{<:Real}
)::Tuple{Vector{Float64}, Vector{Float64}}

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
    a_p, b_p, c_p, P_0, a_t, b_t, T_0 = params
    p_params = [a_p, b_p, c_p, P_0]
    t_params = [a_p, b_p, P_0, a_t, b_t, T_0]

    tspan = (t_start, t_end)

    P_prob = DifferentialEquations.ODEProblem(dpdt, P_0, tspan, p_params)

    P_sol = DifferentialEquations.solve(
        P_prob, 
        DifferentialEquations.Tsit5(); 
        tstops = ts, saveat = ts
    )
    
    Ps = P_sol.u

    # Solve temperature ODE
    T_prob = DifferentialEquations.ODEProblem(dtdt, T_0, tspan, t_params)

    T_sol = DifferentialEquations.solve(
        T_prob, 
        DifferentialEquations.Tsit5(); 
        tstops = ts, saveat = ts
    )

    Ts = T_sol.u

    return Ps, Ts

end


"""Solves the model for a given set of parameters, and returns the sum of 
squared differences between the modelled pressures and temperatures, and the 
data."""
function f_distance(params::Vector{<:Real})::Real

    Ps, Ts = solve_model(params)

    # Calculate the differences between the interpolated pressures and 
    # temperatures, and the data
    P_diffs = sum((((P_obs .- interpolate(ts, Ps, ts_P_obs))) ./ 1e+3) .^2)
    T_diffs = sum((T_obs .- interpolate(ts, Ts, ts_T_obs)).^2)

    return P_diffs + T_diffs

end


"""Plots the pressures and temperatures associated with a given set of model 
parameters against the data."""
function plot_solution(params::Vector{<:Real})::Nothing

    # Solve the model to find the temperatures and pressures associated with 
    # the set of parameters
    Ps, Ts = solve_model(params)

    PyPlot.plot(ts, Ps)
    PyPlot.scatter(ts_P_obs, P_obs)
    PyPlot.savefig("cal_pressures.pdf")
    PyPlot.clf()

    PyPlot.plot(ts, Ts)
    PyPlot.scatter(ts_T_obs, T_obs)
    PyPlot.savefig("cal_temperatures.pdf")
    PyPlot.clf()

    return nothing
    
end

ts_P_obs, P_obs, ts_T_obs, T_obs, ts, qs, dqdts = read_data()

# Specify starting point for parameters
params = [133.056, 0.1160, 1.6905e+9, 1.5600e+6, 1.6416e-4, 8.05e-2, 149.0]

res = Optim.optimize(f_distance, params, Optim.NelderMead())#, Optim.Options(show_trace = true))
println(Optim.converged(res))
cal_params = Optim.minimizer(res)
println(cal_params)

plot_solution(cal_params)


