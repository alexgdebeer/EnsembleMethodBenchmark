struct BumpWell

    cx::Real
    cy::Real
    r::Real

    rates::Tuple
    Z::Real
    
    function BumpWell(
        g::Grid, 
        cx::Real, 
        cy::Real, 
        r::Real,
        rates::Tuple
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
            for (x, y) ∈ zip(g.cxs, g.cys)
                if (r_sq = (x-cx)^2 + (y-cy)^2) < r^2
                    Z += exp(-1/(r^2-r_sq))
                end
            end
        
            return Z
        
        end

        Z = normalising_constant(g, cx, cy, r)

        return new(cx, cy, r, rates, Z)
    
    end

end