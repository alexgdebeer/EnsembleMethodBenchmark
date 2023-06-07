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
    Δ::Function,
    K::AbstractPerturbationKernel,
    T::Int,
    N::Int,
    εs::AbstractVector;
    verbose::Bool = true
)

    θs = Dict(i => [] for i ∈ 1:T)
    ys = Dict(i => [] for i ∈ 1:T)
    ds = Dict(i => [] for i ∈ 1:T)
    ws = Dict(i => [] for i ∈ 1:T)

    for t ∈ 1:T 

        if verbose
            @info("Sampling population $t.")
        end

        if (t > 1) && (typeof(K) <: AbstractAdaptivePerturbationKernel)
            update!(K, θs, ws, ds, t, εs[t])
        end

        i = 0; j = 0

        while i < N

            j += 1

            if t == 1
                θ = sample(π)
            else
                θ⁺ = sample_from_population(θs[t-1], ws[t-1])
                θ = perturb(K, θ⁺, π)
            end

            y_m = f(θ)
            y = G * y_m
            add_noise!(y, e)
            d = Δ(y, y_obs)

            if d ≤ εs[t]
                
                i += 1
                
                if t == 1
                    w = 1.0
                else
                    w = density(π, θ) / 
                        sum(w * density(K, θ⁺, θ) for (θ⁺, w) ∈ zip(θs[t-1], ws[t-1]))
                end

                push!(θs[t], θ)
                push!(ys[t], y_m)
                push!(ds[t], d)
                push!(ws[t], w)
            
            end

            if (verbose) && (j % 1000 == 0)
                α = round(100i/j, digits=2)
                @info("$i / $j sets of parameters accepted (α = $α%).")
            end

        end

        ws[t] ./= sum(ws[t])

    end

    return θs, ys, ws

end


function run_smc_b(
    π::AbstractPrior,
    f::Function,
    y_obs::AbstractVector,
    G::Matrix,
    K::AbstractPerturbationKernel,
    T::Int,
    N::Int,
    Es::AbstractVector;
    verbose::Bool = true
)

    θs = Dict(i => [] for i ∈ 1:T)
    ys = Dict(i => [] for i ∈ 1:T)
    ws = Dict(i => [] for i ∈ 1:T)
    
    for t ∈ 1:T

        if (t > 1) && (typeof(K) <: AbstractAdaptivePerturbationKernel)
            update!(K, θs, ws, t)
        end

        i = 0; j = 0

        while i < N

            j += 1

            if t == 1
                θ = sample(π)
            else
                θ⁺ = sample_from_population(θs[t-1], ws[t-1])
                θ = perturb(K, θ⁺, π)
            end

            y_m = f(θ)
            y = G * y_m

            if (density(Es[t], y - y_obs) / Es[t].c) > rand()
            
                i += 1

                if t == 1
                    w = 1.0
                else
                    w = density(π, θ) / 
                        sum(w * density(K, θ⁺, θ) for (θ⁺, w) ∈ zip(θs[t-1], ws[t-1]))
                end

                push!(θs[t], θ)
                push!(ys[t], y)
                push!(ws[t], w)
            
            end

            if (verbose) && (j % 1000 == 0)
                α = round(100i/j, digits=2)
                @info("$i / $j sets of parameters accepted (α = $α%).")
            end

        end

        # Normalise the weights 
        ws[t] ./= sum(ws[t])

    end

    return θs, ys, ws
    
end


function run_smc(
    π::AbstractPrior,
    f::Function,
    y_obs::AbstractVector,
    G::Matrix,
    K::AbstractPerturbationKernel,
    T::Int,
    N::Int,
    Es::AbstractVector;
    verbose::Bool = true
)

    θs = Dict(i => [] for i ∈ 1:T)
    ys = Dict(i => [] for i ∈ 1:T)
    ws = Dict(i => [] for i ∈ 1:T)
    
    for t ∈ 1:T

        if (t > 1) && (typeof(K) <: AbstractAdaptivePerturbationKernel)
            update!(K, θs, ws, t)
        end

        for i ∈ 1:N

            if t == 1
                θ = sample(π)
            else
                θ⁺ = sample_from_population(θs[t-1], ws[t-1])
                θ = perturb(K, θ⁺, π)
            end

            y = G * f(θ)

            if t == 1
                w = 1.0
            else
                w = (density(π, θ) * density(Es[t], y - y_obs)) / 
                    sum(w * density(K, θ⁺, θ) for (θ⁺, w) ∈ zip(θs[t-1], ws[t-1]))
            end

            push!(θs[t], θ)
            push!(ys[t], y)
            push!(ws[t], w)

            if (verbose) && (i % 1_000 == 0)
                println("Finished sampling $(i) particles.")
            end

        end

        ws[t] ./= sum(ws[t])

    end

    return θs, ys, ws
    
end


function run_abc_mcmc(
    π::AbstractPrior,
    f::Function,
    e::AbstractErrorModel, 
    y_obs::Vector,
    G::Matrix,
    Δ::Function,
    κ::AbstractPerturbationKernel,
    N::Int,
    ε::Real;
    verbose::Bool=true
)

    # Sample a starting point from the prior 
    # θ = sample(π)
    θ = [1.0, 1.0]
    y = f(θ)
    y_m = G * y
    add_noise!(y_m, e)

    while Δ(y_m, y_obs) > ε

        # θ = sample(π)
        θ = [1.0, 1.0]
        y = f(θ)
        y_m = G * y
        add_noise!(y_m, e)

    end

    θs = [θ]
    i = 0

    for j ∈ 2:N

        # Propose a new set of parameters
        θ⁺ = perturb(κ, θ, π)

        # Simulate a dataset
        y = f(θ⁺)
        y_m = G * y
        # add_noise!(y_m, e)
        
        if Δ(y_m, y_obs) ≤ ε

            # Calculate the acceptance probability of θ⁺
            h = (density(π, θ⁺) * density(κ, θ⁺, θ)) / 
                (density(π, θ) * density(κ, θ, θ⁺))
            
            if h ≥ rand()
                i += 1
                θ = copy(θ⁺)
            end
        
        end

        push!(θs, θ)

        if (verbose) && (j % 1000 == 0)
            α = round(100i/j, digits=2)
            @info("$j iterations complete (α = $α%).")
        end

    end

    return θs
    
end


function run_ibis(
    π::AbstractPrior,
    f::Function,
    Ls::AbstractVector,
    y_batches::AbstractVector,
    Gs::AbstractVector,
    N::Int;
    verbose::Bool=true
)

    θs_dict = Dict(i => [] for i ∈ 1:length(y_batches))

    n_batches = length(y_batches)

    # Sample an initial set of particles
    θs = sample(π, n=N)

    ws = ones(N) ./ N
    ys = Vector{Vector{Float64}}(undef, N)

    for j ∈ 1:n_batches  

        for (i, θ) ∈ enumerate(θs) 

            if j == 1
                ys[i] = f(θ)
                ws[i] *= density(Ls[1], Gs[j] * ys[i])
            else
                ws[i] *= density(Ls[j], Gs[j] * ys[i]) / 
                    density(Ls[j-1], Gs[j-1] * ys[i])
            end

        end

        ws ./= sum(ws)

        println(maximum(ws))

        # Re-sample the population with replacement
        θs, ys = SimIntensiveInference.resample_population(θs, ys, ws)
        ws = ones(N) ./ N

        # Generate proposal distribution
        μₖ = Statistics.mean(θs)
        Σₖ = Statistics.cov(θs)

        K = Distributions.MvNormal(μₖ, Σₖ)

        α = 0

        # Perturb the particles using MH kernel
        for (i, θ) ∈ enumerate(θs) 

            θ⁺ = rand(K)
            y⁺ = f(θ⁺)

            h = (density(Ls[j], Gs[j] * y⁺) * density(π, θ⁺) * Distributions.pdf(K, θ)) /
                (density(Ls[j], Gs[j] * ys[i]) * density(π, θ) * Distributions.pdf(K, θ⁺)) 

            if h ≥ rand()
                θs[i] = θ⁺
                ys[i] = y⁺
                α += 1
            end

        end

        if verbose
            @info("Batch $j complete (MH acceptance rate: $(100α/N)).")
        end

        θs_dict[j] = θs

    end

    return θs_dict

end