using Distributions
using Interpolations
using LaTeXStrings
using LinearAlgebra
using Random: seed!
using Statistics

include("../setup/setup.jl")

seed!(16)

ANIMATE = true

# ----------------
# Coarse and fine grid setup
# ----------------

xmin, xmax = 0.0, 1000.0
ymin, ymax = 0.0, 1000.0

Δx_c, Δy_c = 12.5, 12.5
Δx_f, Δy_f = 10.0, 10.0

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

# Indices of timesteps at which well rates change
well_periods = (1, 11, 21) # 0, 40, 80

grid_c = TransientGrid(xmin:Δx_c:xmax, ymin:Δy_c:ymax, tmax, Δt, well_periods, μ, ϕ, c)
grid_f = TransientGrid(xmin:Δx_f:xmax, ymin:Δy_f:ymax, tmax, Δt, well_periods, μ, ϕ, c)

well_radius = 30.0

well_centres = [
    (150, 150), (150, 500), (150, 850),
    (500, 150), (500, 500), (500, 850),
    (850, 150), (850, 500), (850, 850)
]

# Well rates during each time period
well_rates_c = [
    (-q_ps_c, 0, 0), (0, -q_ps_c, 0), (-q_ps_c, 0, 0),
    (0, -q_ps_c, 0), (-q_ps_c, 0, 0), (0, -q_ps_c, 0),
    (-q_ps_c, 0, 0), (0, -q_ps_c, 0), (-q_ps_c, 0, 0)
]

well_rates_f = [
    (-q_ps_f, 0, 0), (0, -q_ps_f, 0), (-q_ps_f, 0, 0),
    (0, -q_ps_f, 0), (-q_ps_f, 0, 0), (0, -q_ps_f, 0),
    (-q_ps_f, 0, 0), (0, -q_ps_f, 0), (-q_ps_f, 0, 0)
]

wells_c = [
    BumpWell(grid_c, cs..., well_radius, qs) 
    for (cs, qs) ∈ zip(well_centres, well_rates_c)
]

wells_f = [
    BumpWell(grid_f, cs..., well_radius, qs) 
    for (cs, qs) ∈ zip(well_centres, well_rates_f)
]

Q_c = sum([w.Q for w ∈ wells_c])
Q_f = sum([w.Q for w ∈ wells_f])

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

us_t = @time solve(grid_f, logps_t, bcs, Q_f)
us_t = reshape(us_t, grid_f.nx, grid_f.ny, grid_f.nt+1)

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

ts_o = [3, 5, 7, 9, 11, 13, 15, 17] # TODO: update

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
    logps = transform(p, θs)
    return solve(grid_c, logps, bcs, Q_c)
end

function F_r(θs::AbstractVector)
    logps = transform(p, θs)
    return solve(grid_c, logps, bcs, Q_c, mu, V_r)
end

function G(us::AbstractVector)
    us = reshape(us, grid_c.nx, grid_c.ny, grid_c.nt+1)
    return get_observations(grid_c, us, ts_o, xs_o, ys_o)
end

# Generate a large number of samples
θs_sample = rand(p, 100)
us_sample = reduce(hcat, [@time F(θ)[:, 2:end] for θ ∈ eachcol(θs_sample)])

mu = vec(mean(us_sample, dims=2))
Γ = cov(us_sample')

eigendecomp = eigen(Γ, sortby=(λ -> -λ))
Λ, V = eigendecomp.values, eigendecomp.vectors

# Extract basis
N_r = findfirst(cumsum(Λ)/sum(Λ) .> 0.999)
V_r = V[:, 1:N_r]

us_sample_r = reduce(hcat, [@time F_r(θ)[:,2:end] for θ ∈ eachcol(θs_sample)])

us_sample_1 = F(θs_sample[:,1])
us_sample_1 = reshape(us_sample_1, grid_c.nx, grid_c.ny, grid_c.nt+1)

us_sample_1r = F_r(θs_sample[:,1])
us_sample_1r = reshape(us_sample_1r, grid_c.nx, grid_c.ny, grid_c.nt+1)

# Different bases for each time period???

@info "Setup complete"

function animate(us, grid, well_inds, fname)

    us ./= 1.0e6
    well_us = us[well_inds...,:]

    anim = @animate for i ∈ axes(us, 3)

        plot(
            heatmap(
                grid.xs, grid.ys, rotl90(us[:,:,i]), 
                clims=extrema(us[2:end-1, 2:end-1, :]), 
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
                title="Pressure in well at (150, 500)",
                legend=:none
            ),
            size=(1000, 400),
            margin=5Plots.mm
        )

    end

    gif(anim, "$fname.gif", fps=4)

end

animate(us_sample_1, grid_c, (13, 41), "darcy_flow")
animate(us_sample_1r, grid_c, (13, 41), "darcy_flow_reduced")