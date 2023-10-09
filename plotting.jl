using LaTeXStrings

function animate(us, grid, well_inds, fname)

    us = reshape(us, grid.nx, grid.nx, :)
    us ./= 1.0e6
    well_us = us[well_inds...,:]

    anim = @animate for i ∈ axes(us, 3)

        plot(
            heatmap(
                grid.xs, grid.xs, us[:, :, i]', 
                clims=extrema(us), 
                cmap=:turbo, 
                size=(500, 500),
                title="Reservoir pressure vs time",
                xlabel=L"x \, \textrm{(m)}",
                ylabel=L"y \, \textrm{(m)}"
            ),
            plot(
                grid.ts[1:i], well_us[1:i], 
                size=(500, 500), 
                xlims=(0, tmax),
                ylims=extrema(well_us),
                xlabel="Day",
                ylabel="Pressure (MPa)",
                title="Pressure in well at (500, 150)",
                legend=:none
            ),
            size=(1000, 400),
            margin=5Plots.mm
        )

    end

    gif(anim, "$fname.gif", fps=4)

end