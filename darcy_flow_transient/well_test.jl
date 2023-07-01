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
    q::Real
    a::Real
    
    function BumpWell(g::Grid, x::Real, y::Real, r::Real, q::Real)

        a = normalising_constant(g, x, y, r)
        return new(x, y, r, q, a)
    
    end

end

struct DeltaWell 

    x::Real 
    y::Real
    q::Real

end

function well_rate(w::BumpWell, x::Real, y::Real)::Real

    r_sq = (x-w.x)^2 + (y-w.y)^2

    if r_sq < w.r^2
        return w.q * exp(-1/(w.r^2-r_sq)) / w.a
    end

    return 0.0

end

function well_rate(w::DeltaWell, x::Real, y::Real)::Real 

    if abs(w.x - x) ≤ 1e-8 && abs(w.y - y) ≤ 1e-8
        return w.q 
    end

    return 0.0
    
end

xmin, Δx, xmax = 0.0, 0.01, 1.0
ymin, Δy, ymax = 0.0, 0.01, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

tmax = 0.2
Δt = 0.001

grid = TimeVaryingGrid(xs, ys, tmax, Δt)

bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => BoundaryCondition(:y0, :neumann, (x, y) -> 0.0), 
    :y1 => BoundaryCondition(:y1, :neumann, (x, y) -> 0.0),
    :t0 => BoundaryCondition(:t0, :initial, (x, y) -> 2.0)
)

wells = [
    BumpWell(grid, 0.2, 0.2, 0.05, -10),
    BumpWell(grid, 0.2, 0.8, 0.05, -10),
    # BumpWell(grid, 0.8, 0.2, 0.05, -10),
    # BumpWell(grid, 0.8, 0.8, 0.05, -10),
    BumpWell(grid, 0.5, 0.5, 0.05, 40)
]

# wells = [
#     DeltaWell(0.2, 0.2, -10),
#     DeltaWell(0.2, 0.8, -10),
#     DeltaWell(0.8, 0.2, -10),
#     DeltaWell(0.8, 0.8, -10),
#     DeltaWell(0.5, 0.5, 40)
# ]

# Define forcing function 
function q(x, y, t)

    q = sum(well_rate(w, x, y) for w ∈ wells)

    return q

end

σ, γx, γy = 1.0, 0.1, 0.1
k = ARDExpSquaredKernel(σ, γx, γy)

μ = 0.0
p = GaussianPrior(μ, k, grid.xs, grid.ys)

logps = reshape(rand(p), grid.nx, grid.ny)
ps = exp.(logps)

us = @time solve(grid, ps, bcs, q)

anim = @animate for i ∈ 1:size(us, 3)

    plot(
        heatmap(
            grid.xs, grid.ys, us[:,:,i]', 
            clims=extrema(us[2:end-1,2:end-1,:]), 
            cmap=:turbo, 
            aspect_ratio=:equal,
            legend=:none
        ),
        axis=([], false), size=(600, 400)
    )

    scatter!([0.2, 0.2, 0.8, 0.8, 0.5], [0.2, 0.8, 0.2, 0.8, 0.5]),

    title!("t = $(grid.ts[i])")

end

gif(anim, "anim.gif", fps=4)