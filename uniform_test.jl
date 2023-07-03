using Distributions
using Statistics

d = Normal()
s = rand(d, 10_000)

s_u = map(s -> cdf(d, s), s)