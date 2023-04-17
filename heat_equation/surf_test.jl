using Plots

x = y = -1:0.1:1;

z = x .^ 2 .+ y' .^ 2;

surface(x, y, z, palette=:balance, zlim=(0, 1))
title!("Test surface")
xlabel!("x")
ylabel!("y")
savefig("test.pdf")