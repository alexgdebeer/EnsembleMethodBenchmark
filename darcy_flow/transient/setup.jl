using Distributions
using Interpolations
using LaTeXStrings
using Random: seed!

include("../setup/setup.jl")

seed!(16)

ANIMATE = true

# ----------------
# Coarse and fine grid setup
# ----------------

xmin, xmax = 0.0, 1000.0
ymin, ymax = 0.0, 1000.0

Δx_c, Δy_c = 20.0, 20.0
Δx_f, Δy_f = 20.0, 20.0

tmax = 120.0
Δt = 4.0

# General parameters
ϕ = 0.30                            # Porosity
μ = 0.5 * 1e-3 / (3600.0 * 24.0)    # Viscosity (Pa⋅day)
c = 2.0e-4 / 6895.0                 # Compressibility (Pa^-1)
u0 = 20 * 1.0e6                     # Initial pressure (Pa)

q_ps_c = 30.0 / (Δx_c * Δy_c)       # Producer rate, (m^3 / day) / m^3
q_is_c = 00.0 / (Δx_c * Δy_c)       # Injector rate, (m^3 / day) / m^3 

q_ps_f = 30.0 / (Δx_f * Δy_f)       # Producer rate, (m^3 / day) / m^3
q_is_f = 00.0 / (Δx_f * Δy_f)       # Injector rate, (m^3 / day) / m^3

grid_c = TransientGrid(xmin:Δx_c:xmax, ymin:Δy_c:ymax, tmax, Δt, μ, ϕ, c)
grid_f = TransientGrid(xmin:Δx_f:xmax, ymin:Δy_f:ymax, tmax, Δt, μ, ϕ, c)

well_r = 30.0

# Define well positions
well_cs = [
    (150, 150), (150, 500), (150, 850),
    (500, 150), (500, 500), (500, 850),
    (850, 150), (850, 500), (850, 850)
]

# Define times wells are active during
well_ts = [
    (00, 40), (40, 80), (00, 40),
    (40, 80), (00, 40), (40, 80),
    (00, 40), (40, 80), (00, 40)
]

wells_c = [
    BumpWell(grid_c, cs..., well_r, ts..., -q_ps_c) 
    for (cs, ts) ∈ zip(well_cs, well_ts)
]

wells_f = [
    BumpWell(grid_f, cs..., well_r, ts..., -q_ps_f) 
    for (cs, ts) ∈ zip(well_cs, well_ts)
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

logp_mu = -14.0
σ_bounds = (0.25, 0.75)
l_bounds = (100, 400)

p = MaternField(grid_c, logp_mu, σ_bounds, l_bounds)

# ----------------
# Truth generation
# ----------------

true_field = MaternField(grid_f, logp_mu, σ_bounds, l_bounds)
θs_t = rand(true_field)
logps_t = transform(true_field, vec(θs_t))
ps_t = 10.0 .^ logps_t

us_t = @time solve(grid_f, ps_t, bcs, q_f)

# ----------------
# Data generation / likelihood
# ----------------

function get_observations(grid, us, ts_o, xs_o, ys_o)

    us_o = []

    for t ∈ ts_o

        u = interpolate((grid.xs, grid.ys), us[:,:,t], Gridded(Linear()))
        append!(us_o, [u(x, y) for (x, y) ∈ zip(xs_o, ys_o)])

    end

    return us_o

end

xs_o = [
    150, 150, 150, 
    500, 500, 500, 
    850, 850, 850
]

ys_o = [
    150, 500, 850, 
    150, 500, 850, 
    150, 500, 850
]

ts_o = [3, 5, 7, 9, 11, 13, 15, 17] # Measure every 10 days for the first 80 days

n_obs = length(xs_o) * length(ts_o)

σ_ϵ = u0 * 0.01
Γ = σ_ϵ^2 * Matrix(1.0I, n_obs, n_obs)

us_o = get_observations(grid_f, us_t, ts_o, xs_o, ys_o)
us_o += rand(MvNormal(Γ))

# L = MvNormal(us_o, Γ_ϵ)

# ----------------
# Model functions
# ----------------

function F(θs::AbstractVector)
    ps = 10.0 .^ transform(p, θs)
    return vec(solve(grid_c, ps, bcs, q_c))
end

function G(us::AbstractVector)
    us = reshape(us, grid_c.nx, grid_c.ny, grid_c.nt+1)
    return get_observations(grid_c, us, ts_o, xs_o, ys_o)
end

@info "Setup complete"

if ANIMATE

    # Rescale pressures and extract pressures at well of interest
    us_t ./= 1.0e6
    well_us = us_t[16,16,:]

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