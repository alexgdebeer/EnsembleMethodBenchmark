using SparseArrays

struct BumpWell

    cx::Real
    cy::Real
    
    qs::Tuple
    Q::AbstractMatrix
    
    function BumpWell(
        g::Grid, 
        cx::Real, 
        cy::Real, 
        r::Real,
        qs::Tuple
    )

        """Calculates the value to scale the bump function by, such that 
        the values of the function on the model grid sum to 1."""
        function normalising_constant(
            g::Grid, 
            cx::Real, 
            cy::Real, 
            r::Real
        )::Real
        
            Z = 0.0
            for (x, y) ∈ zip(g.ixs, g.iys)
                if (r_sq = (x-cx)^2 + (y-cy)^2) < r^2
                    Z += exp(-1/(r^2-r_sq))
                end
            end
        
            return Z
        
        end

        Z = normalising_constant(g, cx, cy, r)
        
        Q_i = Int[]
        Q_j = Int[]
        Q_v = Float64[]

        for (i, (x, y)) ∈ enumerate(zip(g.ixs, g.iys))
            if (dist_sq = (x-cx)^2 + (y-cy)^2) < r^2
                for (j, q) ∈ enumerate(qs)
                    push!(Q_i, i)
                    push!(Q_j, j)
                    push!(Q_v, q * exp(-1/(r^2-dist_sq)) / Z)
                end
            end
        end
        
        Q = sparse(Q_i, Q_j, Q_v, g.nu, length(qs))

        return new(cx, cy, qs, Q)
    
    end

end