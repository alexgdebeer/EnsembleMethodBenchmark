export run_abc, run_abc_smc, run_abc_mcmc, run_mcmc


function run_abc(
    π::AbstractPrior, 
    f::Function,
    e::AbstractErrorModel,
    y_obs::Vector,
    G::Matrix,
    d::Function,
    N::Int,
    α::Real;
    verbose::Bool = true
)

    # Generate some samples of parameters from the prior
    θs = sample(π, n = N)

    # Initialise containers for modelled values and distances
    ys = []
    ds = []

    for (i, θ) ∈ enumerate(θs)

        # Run the forward model
        y = f(θ)

        # Generate noisy modelled values at the points of interest
        y_m = G * y
        add_noise!(y_m, e)

        # Update the variables that store modelled values and distances
        push!(ys, y)
        push!(ds, d(y_m, y_obs))

        if verbose && i % 100 == 0
            @info("Finished running model with $(i) sets of parameters.")
        end

    end
        
    # Filter the α-th quantile sets of modelled values
    ε = Statistics.quantile(ds, α)
    is = findall(≤(ε), ds)

    println("Absolute tolerance: $(ε).")

    return θs, ys, ds, is
        
end


"""Carries out the probabilistic ABC algorithm as described in Wilkinson 
(2013)."""
function run_probabilistic_abc(
    π::AbstractPrior,
    f::Function,
    y_obs::Vector,
    G::Matrix,
    N::Int,
    K::AbstractAcceptanceKernel,
    verbose::Bool = true
)

    θs = sample(π, n = N)

    ys = []
    is = []

    for (i, θ) ∈ enumerate(θs)

        y = f(θ)
        y_m = G * y 
        push!(ys, y)

        #println(θ)
        #println(density(K, y_m - y_obs))

        if density(K, y_m - y_obs) / K.c > rand()
            push!(is, i)
        end

        if verbose && i % 1000 == 0
            @info("Finished running model with $(i) sets of parameters.")
            @info("Acceptance rate: $(length(is) / i).")
        end

    end

end


function run_abc_smc(
    π::AbstractPrior,
    f::Function,
    e::AbstractErrorModel,
    y_obs::Vector,
    G::Matrix,
    d::Function,
    κ::AbstractPerturbationKernel,
    T::Int,
    n::Int,
    α₁::Float64,
    αs::Vector{Float64};
    verbose::Bool = true
)

    θs = Dict(i => [] for i ∈ 1:T)
    ys = Dict(i => [] for i ∈ 1:T)
    ds = Dict(i => [] for i ∈ 1:T)
    is = Dict(i => [] for i ∈ 1:T)
    ws = Dict(i => [] for i ∈ 1:T)

    # Determine how many particles need to be sampled in the first population
    N = round(Int, n / α₁)

    # Run the first population 
    θs[1], ys[1], ds[1], is[1] = run_abc(π, f, e, y_obs, G, d, N, α₁)

    ws[1] = ones(n) ./ n

    # Calculate the acceptance tolerance for the first population 
    ε₁ = maximum(ds[1][is[1]])

    # Define the acceptance tolerances for subsequent populations 
    εs = vcat([ε₁], [α * ε₁ for α ∈ αs])

    for t ∈ 2:T

        println(εs[t])

        # Define the number of accepted particles
        i = 0

        while i < n
            
            # Sample a particle from the previous population and perturb it
            θ⁺ = sample_from_population(θs[t-1][is[t-1]], ws[t-1])
            θ = perturb(κ, θ⁺, π)

            # Run the forward model
            y = f(θ)

            # Generate noisy modelled values at the times of interest
            y_m = G * y
            add_noise!(y_m, e)

            dist = d(y_obs, y_m)

            push!(θs[t], θ)
            push!(ys[t], y)
            push!(ds[t], dist)
            
            if dist ≤ εs[t]

                i += 1

                # Calculate the weight of the particle
                w = density(π, θ) ./ 
                    sum(w * density(κ, θ⁺, θ) 
                        for (θ⁺, w) ∈ zip(θs[t-1][is[t-1]], ws[t-1]))

                push!(is[t], length(ds[t]))
                push!(ws[t], w)

                if verbose && i % 100 == 0
                    @info("$i / $(length(ds[t])) sets of parameters accepted.")
                end

            end

        end

        # Normalise the weights
        ws[t] ./= sum(ws[t])

    end

    return θs, ys, ds, is, ws

end


function run_abc_mcmc(
    π::AbstractPrior,
    f::Function,
    e::AbstractErrorModel, 
    y_obs::Vector,
    G::Matrix,
    d::Function,
    K::AbstractPerturbationKernel,
    N::Int,
    ε::Real
)

    # Sample a starting point from the prior 
    θ = sample(π)
    y = f(θ)
    y_m = G * y
    add_noise!(y_m, e)

    while d(y_m, y_obs) > ε

        θ = sample(π)
        y = f(θ)
        y_m = G * y
        add_noise!(y_m, e)

    end

    # Initialise a vector to contain the chain that is produced
    θs = [θ]

    # Initialise acceptance counter
    α = 0

    for i ∈ 2:N

        # Propose a new set of parameters
        θ⁺ = perturb(K, θ, π)

        # Simulate a dataset
        y = f(θ⁺)

        # Generate noisy modelled values at the times of interest
        y_m = G * y
        add_noise!(y_m, e)
        
        if d(y_m, y_obs) ≤ ε

            # Calculate the acceptance probability of θ⁺
            h = min(
                1, 
                (density(π, θ⁺) * density(K, θ⁺, θ)) / 
                    (density(π, θ) * density(K, θ, θ⁺))
            )
            
            if h ≥ rand()
                α += 1
                θ = θ⁺
            end
        
        end

        push!(θs, θ)

        if (i % 1000 == 0)
            @info("Finished running model with $(i) sets of parameters.")
            println(α / i)
        end 

    end

    return θs, α / N
    
end


function run_mcmc(
    π::AbstractPrior,
    f::Function,
    L::AbstractLikelihood,
    G::Matrix,
    κ::AbstractPerturbationKernel,
    N::Int
)

    θ = sample(π)
    y_m = G * f(θ)

    α = 0

    for i ∈ 1:N 

        # Sample a new parameter 
        θ⁺ = perturb(κ, θ, π)

        # Run the forward model
        y_m⁺ = G * f(θ⁺)

        # Calculate the acceptance probability
        h = min(
            1, 
            (density(π, θ⁺) * density(L, y_m⁺) * density(κ, θ⁺, θ)) / 
                (density(π, θ) * density(L, y_m) * density(κ, θ, θ⁺))
        )

        if h ≥ rand()
            α += 1
            θ = θ⁺
            y_m = y_m⁺
        end

        if i % 100 == 0
            println(α / i)
        end
        
    end

end