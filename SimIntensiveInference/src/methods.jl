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


function run_mcmc(
    f::Function,
    g::Function,
    π::Distribution,
    L::Distribution,
    K::Distribution,
    N::Int;
    θ_s::Union{AbstractVector,Nothing}=nothing,
    verbose::Bool=true
)

    N_θ = length(π.μ)
    N_y = length(L.μ)

    θs = zeros(N_θ, N)
    ys = zeros(N_y, N)

    if θ_s !== nothing
        θs[:,1] = θ_s
    else
        θs[:,1] = rand(π)
    end
    
    ys[:,1] = g(f(θs[:,1]))

    j = 0

    for i ∈ 2:N 

        # Generate a proposal  
        θ_p = θs[:,i-1] + rand(K)

        # Run the forward model 
        y_p = g(f(θ_p))

        # Calculate the acceptance probability
        log_h = @time (logpdf(π, θ_p) + logpdf(L, y_p) + logpdf(K, θs[:,i-1]-θ_p)) -
                (logpdf(π, θs[:,i-1]) + logpdf(L, ys[:,i-1]) + logpdf(K, θ_p-θs[:,i-1]))

        # log_h = (logpdf(π, θ_p) + logpdf(L, y_p)) - (logpdf(π, θs[:,i-1]) + logpdf(L, ys[:,i-1]))

        if log_h ≥ log(rand())
            j += 1
            θs[:,i] = θ_p
            ys[:,i] = y_p
        else
            θs[:,i] = θs[:,i-1]
            ys[:,i] = ys[:,i-1]
        end

        if (verbose) && (i % 1000 == 0)
            α = round(100j/i, digits=2)
            @info("$i iterations complete (α = $α%).")
        end
        
    end

    return θs, ys

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


"""Calculates the MAP estimate of the parameters."""
function calculate_map(
    f::Function, 
    g::Function,
    π::Distribution,
    L::Distribution,
    L_θ::AbstractMatrix,
    L_ϵ::AbstractMatrix
)

    sol = Optim.optimize(
        θ -> sum([L_ϵ*(g(f(θ))-L.μ); L_θ*(θ-π.μ)].^2), 
        collect(π.μ), 
        Optim.Newton(), 
        Optim.Options(show_trace=true, f_abstol=1e-10), 
        autodiff=:forward
    )

    if !Optim.converged(sol) 
        @warn "MAP estimate optimisation failed to converge."
    end

    return Optim.minimizer(sol)

end


function run_rml(
    f::Function,
    g::Function,
    π::Distribution,
    L::Distribution,
    N::Int;
    verbose::Bool=true
)

    N_θ = length(π.μ)
    θs = zeros(N_θ, N)
    
    evals = zeros(N)

    L_θ = cholesky(inv(π.Σ)).U
    L_ϵ = cholesky(inv(L.Σ)).U

    θ_map = calculate_map(f, g, π, L, L_θ, L_ϵ)

    for i ∈ 1:N

        θ_i = rand(π)
        y_i = rand(L)

        res = Optim.optimize(
            θ -> sum([L_ϵ*(g(f(θ))-y_i); L_θ*(θ-θ_i)].^2), 
            θ_map, 
            Optim.Newton(), 
            Optim.Options(show_trace=false, f_abstol=1e-10), 
            autodiff=:forward
        )

        if !Optim.converged(res)
            @warn "MAP estimate optimisation failed to converge."
        end

        θs[:,i] = Optim.minimizer(res)
        evals[i] = Optim.f_calls(res)

        if verbose && i % 100 == 0
            @info "$i samples generated. Mean number of function " *
                "evaluations per optimisation: $(Statistics.mean(evals))."
        end

    end

    return θ_map, θs

end


function run_rto(
    f::Function,
    g::Function,
    π::GaussianPrior,
    L::GaussianLikelihood,
    N::Int;
    verbose::Bool=true
)

    L_θ = LinearAlgebra.cholesky(inv(π.Σ)).U  
    L_ϵ = LinearAlgebra.cholesky(inv(L.Σ)).U

    # Define augmented system 
    f̃(θ) = [L_ϵ*g(f(θ)); L_θ*θ]
    ỹ = [L_ϵ*L.μ; L_θ*π.μ]

    # Calculate the MAP estimate
    res = Optim.optimize(
        θ -> sum((f̃(θ).-ỹ).^2), [1.0, 1.0], 
        Optim.Newton(), 
        Optim.Options(show_trace=false), autodiff=:forward
    )
    
    !Optim.converged(res) && @warn "MAP estimate optimisation failed to converge."
    θ_MAP = Optim.minimizer(res)

    J̃θ_MAP = ForwardDiff.jacobian(f̃, θ_MAP)
    Q = Matrix(LinearAlgebra.qr(J̃θ_MAP).Q)

    θs = []
    ws = []
    evals = []

    for i ∈ 1:N

        # Sample an augmented vector
        ỹ_i = [L_ϵ*sample(L); L_θ*sample(π)]

        # Optimise to find the corresponding set of parameters
        res = Optim.optimize(
            θ -> sum((Q'*(f̃(θ).-ỹ_i)).^2), θ_MAP, 
            Optim.Newton(), 
            Optim.Options(show_trace=false), autodiff=:forward
        )

        !Optim.converged(res) && @warn "Optimisation failed to converge."
        Optim.minimum(res) > 1e-10 && @warn "Non-zero minimum: $(Optim.minimum(res))."
        θ = Optim.minimizer(res)

        J̃θ = ForwardDiff.jacobian(f̃, θ)

        # Compute the weight of the sample
        f̃θ = f̃(θ)
        w = exp(log(abs(LinearAlgebra.det(Q'*J̃θ))^-1) - 0.5sum((f̃θ-ỹ).^2) + 0.5sum((Q'*(f̃θ-ỹ)).^2))

        push!(θs, θ)
        push!(ws, w)
        push!(evals, Optim.f_calls(res))

        if verbose && i % 100 == 0
            @info "$i samples generated. Mean number of function " *
                "evaluations per optimisation: $(Statistics.mean(evals))."
        end

    end

    # Normalise the weights
    ws ./= sum(ws)

    return θ_MAP, Q, θs, ws

end