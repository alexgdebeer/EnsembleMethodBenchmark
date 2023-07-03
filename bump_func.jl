
r_max = 0.05

Δr = r_max / 100.0

xs = -r_max:Δr:r_max
ys = -r_max:Δr:r_max

nx = length(xs)
ny = length(ys)

Δx = xs[2] - xs[1]
Δy = ys[2] - ys[1]

zs = zeros(nx, ny)

for i ∈ 1:nx, j ∈ 1:ny

    r_sq = xs[i]^2 + ys[j]^2

    if r_sq < r_max^2
        zs[i, j] = exp(-1/(r_max^2-r_sq))
        if zs[i, j] > 1.0
            println(zs[i, j])
        end
    end

end

a = Δx * Δy * sum(zs)
zs ./= a