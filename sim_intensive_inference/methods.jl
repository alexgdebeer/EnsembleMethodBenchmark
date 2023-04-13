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
    π::AbstractPrior,
    f::Function,
    L::AbstractLikelihood,
    G::Matrix,
    κ::AbstractPerturbationKernel,
    N::Int;
    verbose::Bool = true
)

    # θ = sample(π)
    θ = [1.0, 1.0]

    y_m = G * f(θ)

    θs = [θ]
    i = 0

    for j ∈ 2:N 

        # Sample a new parameter 
        θ⁺ = perturb(κ, θ, π)

        # Run the forward model
        y_m⁺ = G * f(θ⁺)

        # Calculate the acceptance probability
        h = (density(π, θ⁺) * density(L, y_m⁺) * density(κ, θ⁺, θ)) / 
                (density(π, θ) * density(L, y_m) * density(κ, θ, θ⁺))

        if h ≥ rand()
            i += 1
            θ = copy(θ⁺)
            y_m = copy(y_m⁺)
        end

        push!(θs, θ)

        if (verbose) && (j % 1000 == 0)
            α = round(100i/j, digits=2)
            @info("$j iterations complete (α = $α%).")
        end
        
    end

    return θs

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


"""
Draws N samples from the approximate posterior using the randomised maximum 
likelihood algorithm.

TODO: find a way to save the function evaluated at each value of θ.
TODO: consider changing the optimiser / allowing for a Jacobian to be passed in.
TODO: check for convergence to the MAP estimate?
"""
function run_rml(
    f::Function,
    π::GaussianPrior,
    L::GaussianLikelihood,
    G::Matrix,
    N::Int;
    verbose::Bool=true
)

    L_p = LinearAlgebra.cholesky(inv(π.Σ)).U  
    L_y = LinearAlgebra.cholesky(inv(L.Σ)).U

    # Calculate the MAP estimate
    map_func(θ) = 0.5sum([L_p*(θ-π.μ); L_y*(G*f(θ)-L.μ)].^2)
    res = Optim.optimize(map_func, [1.0, 1.0], Optim.NelderMead())
    θ_MAP = Optim.minimizer(res)

    println(θ_MAP)

    # TODO: check for convergence?

    θs = []

    for i ∈ 1:N

        θ⁺ = sample(π)
        y⁺ = sample(L)

        # TODO: consider changing the optimiser here (could use LBFGS?)
        θ_func(θ) = 0.5sum([L_p*(θ-θ⁺); L_y*(G*f(θ)-y⁺)].^2)
        res = Optim.optimize(θ_func, θ_MAP, Optim.NelderMead())

        push!(θs, Optim.minimizer(res))

        if verbose && i % 100 == 0
            @info("$i samples generated.")
        end

    end

    return θ_MAP, θs

end


function run_rto(
    f::Function,
    π::GaussianPrior,
    L::GaussianLikelihood,
    G::Matrix,
    N::Int;
    verbose::Bool=true
)

    L_θ = LinearAlgebra.cholesky(inv(π.Σ)).U  
    L_ϵ = LinearAlgebra.cholesky(inv(L.Σ)).U

    # Define augmented system 
    f̃(θ) = [L_ϵ*G*f(θ); L_θ*θ]
    ỹ = [L_ϵ*L.μ; L_θ*π.μ]

    # Calculate the MAP estimate
    map_func(θ) = 0.5sum((f̃(θ)-ỹ).^2)
    res = Optim.optimize(map_func, [1.0, 1.0], Optim.NelderMead())
    θ_MAP = Optim.minimizer(res)

    J̃θ_MAP = ForwardDiff.jacobian(f̃, θ_MAP)
    Q = Matrix(LinearAlgebra.qr(J̃θ_MAP))
    LinearAlgebra.normalize!.(eachcol(Q))

    θs = []
    ws = []
    evals = []

    for i ∈ 1:N

        ỹⁱ = [L_ϵ*sample(L); L_θ*sample(π)]

        θ_func(θ) = sum((Q' * (f̃(θ)-ỹⁱ)).^2)
        res = Optim.optimize(θ_func, θ_MAP, Optim.NelderMead())

        Optim.minimum(res) > 1e-6 && @warn "Non-zero result of optimisation."

        θ = Optim.minimizer(res)
        J̃θ = ForwardDiff.jacobian(f̃, θ)

        fθ = G*f(θ)
        f̃θ = [L_ϵ*fθ; L_θ*θ]

        w = abs(LinearAlgebra.det(Q' * J̃θ))^-1 * 
            exp(-0.5sum((f̃θ-ỹ).^2) + 0.5sum((Q'*(f̃θ-ỹ)).^2))
        
        push!(θs, θ)
        push!(ws, w)
        push!(evals, Optim.f_calls(res))

        if verbose && i % 100 == 0
            @info "$i samples generated. Mean number of function " *
                "evaluations per optimisation: $(Statistics.mean(evals))."
        end

    end

    ws ./= sum(ws)

    return θ_MAP, θs, ws

end


function run_enkf(
    f::Function,
    g::Function,
    ts_obs::Vector, 
    ys_obs::Vector,
    σ_y::Real,
    π_θ::AbstractPrior,
    π_p::AbstractPrior,
    N_e::Int
)

    # Define a vector that offsets the times by 1
    ts_obs_p = [0.0, ts_obs[1:(end-1)]...]

    # Generate a number of initial samples
    θs = sample(π_θ, n=N_e)
    ps = sample(π_p, n=N_e)

    n_θs = length(π_θ.μ)
    
    # Run the samples forward to the first time at which measurements were recorded.
    for (t_p, t, ys) ∈ zip(ts_obs_p, ts_obs, ys_obs)
        
        # Run the state model and measurement model for each ensemble member
        p̃s = [f(θ; x₀=p, t_start=t_p, t_end=t)[[Int(end/2), end]] for (θ, p) ∈ zip(θs, ps)]
        ũs = [vcat(θ, p̃) for (θ, p̃) ∈ zip(θs, p̃s)]
        g̃s = [g(θ, p̃) for (θ, p̃) ∈ zip(θs, p̃s)]

        # Generate some pertubed data
        Γ_y = σ_y^2 * Matrix(1.0LinearAlgebra.I, length(ys), length(ys))
        ỹs = [rand(Distributions.MvNormal(ys, Γ_y)) for _ ∈ 1:N_e]

        # Compute centred matrices for the vectors of states and predictions
        ũs_c = hcat(ũs...)'; ũs_c .-= Statistics.mean(ũs_c, dims=1)
        g̃s_c = hcat(g̃s...)'; g̃s_c .-= Statistics.mean(g̃s_c, dims=1)

        Γ_ug = (ũs_c'*g̃s_c) / (N_e-1.0)
        Γ_gg = (g̃s_c'*g̃s_c) / (N_e-1.0)

        # Calculate the gain matrix 
        K = Γ_ug * inv(Γ_gg + Γ_y)

        # Update each ensemble member 
        us = [ũ + K*(ỹ-g̃) for (ũ, ỹ, g̃) ∈ zip(ũs, ỹs, g̃s)]

        # Extract the updated parameters and observations
        θs = [u[1:n_θs] for u ∈ us]
        ps = [u[(n_θs+1):end] for u ∈ us]

    end

    println(θs)

    return θs

end