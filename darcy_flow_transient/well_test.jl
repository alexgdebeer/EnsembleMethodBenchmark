using Random: seed!
using SimIntensiveInference
include("setup/setup.jl")

seed!(16)

function normalising_constant(g::Grid, x::Real, y::Real, r::Real)::Real

    a = 0.0

    for i ∈ 1:g.nx, j ∈ 1:g.ny

        r_sq = (g.xs[i] - x)^2 + (g.ys[j] - y)^2
    
        if r_sq < r^2
            a += exp(-1/(r^2-r_sq))
        end
    
    end

    return a

end

struct BumpWell

    x::Real
    y::Real
    r::Real
    t0::Real
    t1::Real
    q::Real
    a::Real
    
    function BumpWell(
        g::Grid, 
        x::Real, 
        y::Real, 
        r::Real,
        t0::Real,
        t1::Real, 
        q::Real
    )

        a = normalising_constant(g, x, y, r)
        return new(x, y, r, t0, t1, q, a)
    
    end

end

struct DeltaWell 

    x::Real 
    y::Real
    t0::Real 
    t1::Real
    q::Real

end

function well_rate(w::BumpWell, x::Real, y::Real, t::Real)::Real

    if t < w.t0 || t > w.t1
        return 0.0
    end

    r_sq = (x-w.x)^2 + (y-w.y)^2

    if r_sq ≥ w.r^2
        return 0.0
    end

    return w.q * exp(-1/(w.r^2-r_sq)) / w.a

end

function well_rate(w::DeltaWell, x::Real, y::Real, t::Real)::Real 

    if t < w.t0 || t > w.t1
        return 0.0
    end

    if abs(w.x - x) ≤ 1e-8 && abs(w.y - y) ≤ 1e-8
        return w.q 
    end

    return 0.0
    
end

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

grid = TimeVaryingGrid(xs, ys, tmax, Δt, μ, ϕ, c)

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

# Define forcing function 
q(x, y, t) = sum(well_rate(w, x, y, t) for w ∈ wells)

σ, γx, γy = 0.5, 100, 100
k = ARDExpSquaredKernel(σ, γx, γy)

logμ = -14.0
p = GaussianPrior(logμ, k, grid.xs, grid.ys)

logps = reshape(rand(p), grid.nx, grid.ny)
ps = 10.0 .^ logps

us = @time solve(grid, ps, bcs, q)
us ./= 1.0e6

anim = @animate for i ∈ axes(us, 3)

    plot(
        heatmap(
            grid.xs, grid.ys, us[:,:,i]', 
            clims=extrema(us[2:end-1,2:end-1,:]), 
            cmap=:turbo, 
            aspect_ratio=:equal,
            colorbar_title="Pressure",
            title="t = $(grid.ts[i])",
            titlefontsize=32,
            colorbar_titlefontsize=20,
            tickfontsize=16
            # legend=:none
        ),
        #axis=([], false), 
        size=(700, 700)
    )

end

gif(anim, "anim.gif", fps=4)