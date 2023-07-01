using SimIntensiveInference
include("setup/setup.jl")

function normalising_constant(r_max::Real)::Real

    Δr = r_max / 100.0

    xs = -r_max:Δr:r_max
    ys = -r_max:Δr:r_max

    nx = length(xs)
    ny = length(ys)

    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]

    z_sum = 0.0

    for i ∈ 1:nx, j ∈ 1:ny

        r_sq = xs[i]^2 + ys[j]^2
    
        if r_sq < r_max^2
            z_sum += exp(-1/(r_max^2-r_sq))
        end
    
    end

    a = Δx * Δy * z_sum

    return a

end

struct BumpWell

    x::Real
    y::Real
    r::Real

    q::Real

    a::Real
    
    function BumpWell(x::Real, y::Real, r::Real, q::Real)

        a = normalising_constant(r)
        return new(x, y, r, q, a)
    
    end

end

function well_rate(b::BumpWell, x::Real, y::Real)::Real

    r_sq = (x-b.x)^2 * (y-b.y)^2

    if r_sq < b.r^2
        return b.q * exp(-1/(b.r^2-r_sq)) / b.a
    end

    return 0.0

end

xmin, Δx, xmax = 0.0, 0.02, 1.0
ymin, Δy, ymax = 0.0, 0.02, 1.0

xs = xmin:Δx:xmax
ys = ymin:Δy:ymax

tmax = 2.0
Δt = 0.01

grid = TimeVaryingGrid(xs, ys, tmax, Δt)

bcs = Dict(
    :x0 => BoundaryCondition(:x0, :neumann, (x, y) -> 0.0), 
    :x1 => BoundaryCondition(:x1, :neumann, (x, y) -> 0.0),
    :y0 => BoundaryCondition(:y0, :neumann, (x, y) -> 0.0), 
    :y1 => BoundaryCondition(:y1, :neumann, (x, y) -> 0.0),
    :t0 => BoundaryCondition(:t0, :initial, (x, y) -> 2.0)
)

wells = [
    BumpWell(0.2, 0.2, 0.05, -1e-6),
    BumpWell(0.2, 0.8, 0.05, -1e-6),
    BumpWell(0.8, 0.2, 0.05, -1e-6),
    BumpWell(0.8, 0.8, 0.05, -1e-6),
    BumpWell(0.5, 0.5, 0.05, 4e-6)
]

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
        heatmap(us[2:end-1,2:end-1,i]', cmap=:turbo, aspect_ratio=:equal),
        axis=([], false), size=(400, 400)
    )

    title!("t = $(grid.ts[i])")

end

gif(anim, "anim.gif", fps=4)