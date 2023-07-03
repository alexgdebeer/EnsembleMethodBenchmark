using Distributions
using Interpolations
using LaTeXStrings
using Random: seed!
using SimIntensiveInference

include("../setup/setup.jl")

seed!(16)

animate = true

# ----------------
# Coarse and fine grid setup
# ----------------

xmin, xmax = 0.0, 1000.0
ymin, ymax = 0.0, 1000.0

Δx_c, Δy_c = 50.0, 50.0
Δx_f, Δy_f = 50.0, 50.0

tmax = 90.0
Δt = 2.0

# General parameters
ϕ = 0.20                            # Porosity
μ = 5.0e-4 / (3600.0 * 24.0)        # Viscosity (Pa⋅day)
c = 1.0e-8                          # Compressibility (Pa^-1)
u0 = 2.0e7                          # Initial pressure (Pa)

grid_c = TransientGrid(xmin:Δx_c:xmax, ymin:Δy_c:ymax, tmax, Δt, μ, ϕ, c)
grid_f = TransientGrid(xmin:Δx_f:xmax, ymin:Δy_f:ymax, tmax, Δt, μ, ϕ, c)

q_ps_c = 20.0 / (Δx_c * Δy_c)       # Producer rate, (m^3 / day) / m^3
q_is_c = 00.0 / (Δx_c * Δy_c)       # Injector rate, (m^3 / day) / m^3 

q_ps_f = 20.0 / (Δx_f * Δy_f)       # Producer rate, (m^3 / day) / m^3
q_is_f = 00.0 / (Δx_f * Δy_f)       # Injector rate, (m^3 / day) / m^3 

wells_c = [
    BumpWell(grid_c, 200, 200, 30, 0, 30, -q_ps_c),
    BumpWell(grid_c, 200, 800, 30, 30, 60, -q_ps_c),
    BumpWell(grid_c, 800, 800, 30, 0, 30, -q_ps_c),
    BumpWell(grid_c, 800, 200, 30, 30, 60, -q_ps_c),
    BumpWell(grid_c, 500, 500, 30, 0, 60, -q_ps_c)
]

wells_f = [
    BumpWell(grid_f, 200, 200, 30, 00, 30, -q_ps_f),
    BumpWell(grid_f, 200, 500, 30, 00, 30, -q_ps_f),
    BumpWell(grid_f, 200, 800, 30, 00, 30, -q_ps_f),
    BumpWell(grid_f, 500, 200, 30, 30, 60, -q_ps_f),
    BumpWell(grid_f, 500, 500, 30, 30, 60, -q_ps_f),
    BumpWell(grid_f, 500, 800, 30, 30, 60, -q_ps_f),
    BumpWell(grid_f, 800, 200, 30, 60, 90, -q_ps_f),
    BumpWell(grid_f, 800, 500, 30, 60, 90, -q_ps_f),
    BumpWell(grid_f, 800, 800, 30, 60, 90, -q_ps_f)
]

q_c(x, y, t) = sum(well_rate(w, x, y, t) for w ∈ wells_c)
q_f(x, y, t) = sum(well_rate(w, x, y, t) for w ∈ wells_f)

bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => BoundaryCondition(:y0, :neumann, (x, y) -> 0.0), 
    :y1 => BoundaryCondition(:y1, :neumann, (x, y) -> 0.0),
    :t0 => BoundaryCondition(:t0, :initial, (x, y) -> u0)
)

# ----------------
# Prior generation
# ----------------

logμ_p = -14.0
σ_p = 0.2
γx_p, γy_p = 100, 100
k = ARDExpSquaredKernel(σ_p, γx_p, γy_p)

p = GaussianPrior(logμ_p, k, grid_c.xs, grid_c.ys)

# ----------------
# Truth generation
# ----------------

logps_t = rand(GaussianPrior(logμ_p, k, grid_f.xs, grid_f.ys))
logps_t = reshape(logps_t, grid_f.nx, grid_f.ny)
ps_t = 10.0 .^ logps_t

us_t = @time solve(grid_f, ps_t, bcs, q_f)

# ----------------
# Data generation / likelihood
# ----------------

function get_observations(grid, us, ts_o, xs_o, ys_o)

    us_o = []

    for t ∈ ts_o

        u = interpolate((grid.xs, grid.ys), us[:,:,t], Gridded(Linear()))
        push!(us_o, [u(x, y) for x ∈ xs_o for y ∈ ys_o]...)

    end

    return us_o

end

xs_o = [200, 500, 800]
ys_o = [200, 500, 800]
ts_o = [6, 11, 16, 21]
n_obs = length(xs_o) * length(ys_o) * length(ts_o)

σ_ϵ = u0 * 0.01
Γ_ϵ = σ_ϵ^2 * Matrix(1.0I, n_obs, n_obs)

us_o = get_observations(grid_f, us_t, ts_o, xs_o, ys_o)
us_o += rand(MvNormal(Γ_ϵ))

L = MvNormal(us_o, Γ_ϵ)

# ----------------
# Model functions
# ----------------

function f(logps::AbstractVector)
    ps = 10.0 .^ reshape(logps, grid_c.nx, grid_c.ny)
    return solve(grid_c, ps, bcs, q_c)
end

function g(us::AbstractArray)
    return get_observations(grid_c, us, ts_o, xs_o, ys_o)
end

@info "Setup complete"

if animate

    # Rescale pressures and extract pressures at well of interest
    us_t ./= 1.0e6
    well_us = us_t[21,21,:]

    anim = @animate for i ∈ axes(us_t, 3)

        plot(
            heatmap(
                grid_f.xs, grid_f.ys, us_t[:,:,i]', 
                clims=extrema(us_t[2:end-1, 2:end-1, :]), 
                cmap=:turbo, 
                size=(500, 500),
                title="Reservoir pressure vs time",
                xlabel=L"x \, \textrm{(m)}",
                ylabel=L"y \, \textrm{(m)}"
            ),
            plot(
                grid_f.ts[1:i], well_us[1:i], 
                size=(500, 500), 
                xlims=(0, tmax),
                ylims=extrema(well_us),
                xlabel="Day",
                ylabel="Pressure (MPa)",
                title="Pressure in well at (200, 200)",
                legend=:none
            ),
            size=(1000, 400),
            margin=5Plots.mm
        )

    end

    gif(anim, "pressure_plots.gif", fps=4)

end