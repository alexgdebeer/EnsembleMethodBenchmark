using LaTeXStrings
using Random: seed!
using SimIntensiveInference

include("../setup/setup.jl")

seed!(16)

animate = true

xmin, Δx, xmax = 0.0, 10.0, 1000.0
ymin, Δy, ymax = 0.0, 10.0, 1000.0
h = 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

tmax = 60.0
Δt = 1.0

# General parameters
ϕ = 0.3                         # Porosity
μ = 5.0e-4 / (3600.0 * 24.0)    # Viscosity, Pa⋅day
c = 1.0e-8                      # Compressibility, Pa^-1
u0 = 2.0e7                      # Initial pressure, Pa

q_ps = 30.0 / (Δx * Δy * h)     # Producer rate, (m^3 / day) / m^3
q_is = 0.0 / (Δx * Δy * h)      # Injector rate, (m^3 / day) / m^3 

grid = TransientGrid(xs, ys, tmax, Δt, μ, ϕ, c)

bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => BoundaryCondition(:y0, :neumann, (x, y) -> 0.0), 
    :y1 => BoundaryCondition(:y1, :neumann, (x, y) -> 0.0),
    :t0 => BoundaryCondition(:t0, :initial, (x, y) -> u0)
)

wells = [
    BumpWell(grid, 200, 200, 30, 0, 30, -q_ps),
    BumpWell(grid, 200, 800, 30, 30, 60, -q_ps),
    BumpWell(grid, 800, 800, 30, 0, 30, -q_ps),
    BumpWell(grid, 800, 200, 30, 30, 60, -q_ps),
    BumpWell(grid, 500, 500, 30, 0, 60, q_is)
]

# wells = [
#     DeltaWell(200, 200, 0, 30, -q_ps),
#     DeltaWell(200, 800, 30, 60, -q_ps),
#     DeltaWell(800, 800, 0, 30, -q_ps),
#     DeltaWell(800, 200, 30, 60, -q_ps),
#     DeltaWell(500, 500, 0, 60, q_is)
# ]

q(x, y, t) = sum(well_rate(w, x, y, t) for w ∈ wells)

σ, γx, γy = 0.5, 100, 100
k = ARDExpSquaredKernel(σ, γx, γy)

logμ = -14.0
p = GaussianPrior(logμ, k, grid.xs, grid.ys)

logps = reshape(rand(p), grid.nx, grid.ny)
ps = 10.0 .^ logps

us = @time solve(grid, ps, bcs, q)

if animate

    # Rescale pressures and extract pressures at well of interest
    us ./= 1.0e6
    well_us = us[21,22,:]

    anim = @animate for i ∈ axes(us, 3)

        plot(
            heatmap(
                grid.xs, grid.ys, us[:,:,i]', 
                clims=extrema(us[2:end-1, 2:end-1, :]), 
                cmap=:turbo, 
                size=(500, 500),
                title="Reservoir pressure vs time",
                xlabel=L"x \, \textrm{(m)}",
                ylabel=L"y \, \textrm{(m)}"
            ),
            plot(
                well_us[1:i], 
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