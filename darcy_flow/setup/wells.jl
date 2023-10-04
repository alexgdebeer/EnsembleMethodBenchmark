"""Calculates the value to scale a bump function by, such that the 
values of the function on a grid sum to 1."""
function normalising_constant(g::Grid, x::Real, y::Real, r::Real)::Real

    a = 0.0

    for gx ∈ g.xs, gy ∈ g.ys

        r_sq = (gx - x)^2 + (gy - y)^2
    
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
    qs::Tuple
    a::Real
    
    function BumpWell(
        g::Grid, 
        x::Real, 
        y::Real, 
        r::Real,
        qs::Tuple
    )

        a = normalising_constant(g, x, y, r)
        return new(x, y, r, qs, a)
    
    end

end

function well_rate(w::BumpWell, x::Real, y::Real, p::Int)::Real

    r_sq = (x - w.x)^2 + (y - w.y)^2

    if r_sq ≥ w.r^2
        return 0.0
    end

    return w.qs[p] * exp(-1/(w.r^2-r_sq)) / w.a

end