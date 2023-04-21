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


function run_rml(
    f::Function,
    g::Function,
    π::GaussianPrior,
    L::GaussianLikelihood,
    N::Int;
    verbose::Bool=true
)

    L_ϵ = LinearAlgebra.cholesky(inv(L.Σ)).U
    L_θ = LinearAlgebra.cholesky(inv(π.Σ)).U  

    # Calculate the MAP estimate
    res = Optim.optimize(
        θ -> sum([L_ϵ*(g(f(θ))-L.μ); L_θ*(θ-π.μ)].^2), [1.0, 1.0], 
        Optim.Newton(), 
        Optim.Options(show_trace=false, f_abstol=1e-10), autodiff=:forward
    )

    θ_MAP = Optim.minimizer(res)
    !Optim.converged(res) && @warn "MAP estimate optimisation failed to converge."

    θs = []
    evals = []

    for i ∈ 1:N

        θ_i = sample(π)
        y_i = sample(L)

        res = Optim.optimize(
            θ -> sum([L_ϵ*(g(f(θ)).-y_i); L_θ*(θ.-θ_i)].^2), θ_MAP, 
            Optim.Newton(), 
            Optim.Options(show_trace=false, f_abstol=1e-10), autodiff=:forward
        )

        !Optim.converged(res) && @warn "MAP estimate optimisation failed to converge."

        push!(θs, Optim.minimizer(res))
        push!(evals, Optim.f_calls(res))

        if verbose && i % 100 == 0
            @info "$i samples generated. Mean number of function " *
                "evaluations per optimisation: $(Statistics.mean(evals))."
        end

    end

    return θ_MAP, θs

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


function run_enkf(
    f::Function,
    H::Function,
    π_u::SimIntensiveInference.AbstractPrior,
    ts::Vector,
    ys::Matrix,
    θs::Vector,
    t_1::Real,
    σ_ϵ::Real,
    N_e::Int
)

    # Generate an initial ensemble by sampling from the prior
    us_e = SimIntensiveInference.sample(π_u, n=N_e)

    # Define a vector that offsets the times by 1
    ts_0 = [0.0, ts[1:(end-1)]...]

    us_e_combined = reduce(vcat, us_e)
    us_e = reduce(hcat, us_e)

    for (t_0, t_1, y) ∈ zip(ts_0, ts, eachcol(ys))

        # Run each ensemble member forward in time 
        us_e = [f(θs; y_0=u, t_0=t_0, t_1=t_1)[:, 2:end] for u ∈ eachcol(us_e)]
        us_e_combined = hcat(us_e_combined, reduce(vcat, us_e))

        # Extract the forecast states and generate the predictions 
        us_ef = reduce(hcat, [u[:, end] for u ∈ us_e])
        ys_ef = H(us_ef, θs)

        # Generate a set of perturbed data vectors 
        Γ_ϵϵ = σ_ϵ^2 * Matrix(LinearAlgebra.I, length(y), length(y))
        ys_p = rand(Distributions.MvNormal(y, Γ_ϵϵ), N_e)

        # Compute the Kalman gain
        U_c = us_ef * (I - ones(N_e, N_e)/N_e)
        Y_c = ys_ef * (I - ones(N_e, N_e)/N_e)
        Γ_uy_e = 1/(N_e-1)*U_c*Y_c'
        Γ_yy_e = 1/(N_e-1)*Y_c*Y_c'
        K = Γ_uy_e * inv(Γ_yy_e + Γ_ϵϵ)
        
        us_e = us_ef + K*(ys_p-ys_ef)

    end

    # Run each ensemble member to the final timestep if necessary
    if ts[end] < t_1
        us_e = [f(θs; y_0=u, t_0=ts[end])[:, 2:end] for u ∈ eachcol(us_e)]
        us_e_combined = hcat(us_e_combined, reduce(vcat, us_e))
    end

    return us_e_combined

end


"""Runs the EnKF algorithm, with the parameters augmented to the states."""
function run_enkf_params(
    f::Function,
    H::Function,
    π_u::SimIntensiveInference.AbstractPrior,
    π_θ::SimIntensiveInference.AbstractPrior,
    ts::Vector,
    ys::Matrix,
    t_1::Real,
    σ_ϵ::Real,
    N_e::Int
)

    # Define a vector that offsets the times by 1
    ts_0 = [0.0, ts[1:(end-1)]...]

    n_us = length(π_u.μ)

    # Generate an initial sample of states and parameters from the prior
    us_e = sample(π_u, n=N_e)
    θs_e = sample(π_θ, n=N_e)

    us_e_combined = reduce(vcat, us_e)
    θs_e_combined = reduce(vcat, θs_e)

    us_e = reduce(hcat, us_e)
    θs_e = reduce(hcat, θs_e)

    for (t_0, t_1, y) ∈ zip(ts_0, ts, eachcol(ys))

        # Run each ensemble member forward in time 
        us_e = [f(θ; y_0=u, t_0=t_0, t_1=t_1)[:, 2:end] 
                    for (u, θ) ∈ zip(eachcol(us_e), eachcol(θs_e))]
        
        us_e_combined = hcat(us_e_combined, reduce(vcat, us_e))
        θs_e_combined = hcat(θs_e_combined, reduce(vcat, θs_e))

        # Extract the forecast states and generate the predictions 
        us_ef = reduce(hcat, [u[:, end] for u ∈ us_e])
        ys_ef = H(us_ef, θs_e)

        # Generate a set of perturbed data vectors 
        Γ_ϵϵ = σ_ϵ^2 * Matrix(LinearAlgebra.I, length(y), length(y))
        ys_p = rand(Distributions.MvNormal(y, Γ_ϵϵ), N_e)

        # Compute the Kalman gain
        U_c = vcat(us_ef, θs_e) * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Y_c = ys_ef * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Γ_uy_e = 1/(N_e-1)*U_c*Y_c'
        Γ_yy_e = 1/(N_e-1)*Y_c*Y_c'
        K = Γ_uy_e * inv(Γ_yy_e + Γ_ϵϵ)
        
        uθs_e = vcat(us_ef, θs_e) + K*(ys_p-ys_ef)
        us_e = uθs_e[1:n_us, :]
        θs_e = uθs_e[(n_us+1):end, :]

    end

    # Run each ensemble member to the final timestep if necessary
    if ts[end] < t_1
        us_e = [f(θ; y_0=u, t_0=ts[end])[:, 2:end] 
                    for (u, θ) ∈ zip(eachcol(us_e), eachcol(θs_e))]
        us_e_combined = hcat(us_e_combined, reduce(vcat, us_e))
    end

    return us_e_combined, θs_e_combined

end


function run_enkf_simplified(
    H::Function,
    π::AbstractPrior,
    ts::AbstractVector,
    ys::AbstractMatrix,
    σ_ϵ::Real,
    N_e::Int;
    verbose::Bool=true
)

    # Sample an ensemble of sets of parameters from the prior
    θs_e = reduce(hcat, sample(π, n=N_e))

    for (i, (t, y)) ∈ enumerate(zip(ts, eachcol(ys)))

        # Generate the ensemble predictions for the current time
        ys_e = reduce(hcat, [H(θ, t) for θ ∈ eachcol(θs_e)])

        # Generate a set of perturbed data vectors 
        Γ_ϵϵ = σ_ϵ^2 * Matrix(LinearAlgebra.I, length(y), length(y))
        ys_p = rand(Distributions.MvNormal(y, Γ_ϵϵ), N_e)

        # Compute the Kalman gain
        θ_c = θs_e * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Y_c = ys_e * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Γ_θy_e = 1/(N_e-1)*θ_c*Y_c'
        Γ_yy_e = 1/(N_e-1)*Y_c*Y_c'
        K = Γ_θy_e * inv(Γ_yy_e + Γ_ϵϵ)

        # Update each ensemble member
        θs_e = θs_e + K*(ys_p-ys_e)

        println(Γ_yy_e)

        verbose && @info "Iteration $i complete."

    end

    return θs_e

end


function run_ensemble_smoother(
    f::Function,
    g::Function,
    π::AbstractPrior,
    ys::AbstractVector,
    σ_ϵ::Real, 
    N_e::Int;
    verbose::Bool=true
)

    # Sample an ensemble from the prior
    θs_e = reduce(hcat, sample(π, n=N_e))

    # Generate the ensemble predictions
    ys_e = reduce(hcat, [g(f(θ)) for θ ∈ eachcol(θs_e)])

    # Generate a set of perturbed data vectors 
    Γ_ϵ = σ_ϵ^2 * Matrix(LinearAlgebra.I, length(ys), length(ys))
    ys_p = rand(Distributions.MvNormal(ys, Γ_ϵ), N_e)

    # Compute the gain
    θ_c = θs_e * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
    Y_c = ys_e * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
    Γ_θy_e = 1/(N_e-1)*θ_c*Y_c'
    Γ_y_e = 1/(N_e-1)*Y_c*Y_c'
    K = Γ_θy_e * inv(Γ_y_e + Γ_ϵ)

    # Update each ensemble member
    θs_e = θs_e + K*(ys_p-ys_e)

    return θs_e

end


function run_ensemble_smoother_mda(
    f::Function,
    g::Function,
    π::AbstractPrior,
    ys::AbstractVector,
    σ_ϵ::Real, 
    αs::AbstractVector,
    N_e::Int;
    verbose::Bool=true
)

    if abs(sum(1 ./ αs) - 1.0) > 1e-4 
        error("Reciprocals of α values do not sum to 1.")
    end

    # Sample an ensemble from the prior
    θs_e = reduce(hcat, sample(π, n=N_e))

    for α ∈ αs 

        # Generate the ensemble predictions 
        ys_e = reduce(hcat, [g(f(θ)) for θ ∈ eachcol(θs_e)])

        # Generate a set of perturbed data vectors 
        Γ_ϵϵ = α * σ_ϵ^2 * Matrix(LinearAlgebra.I, length(ys), length(ys))
        ys_p = rand(Distributions.MvNormal(ys, Γ_ϵϵ), N_e)

        # Compute the gain
        θ_c = θs_e * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Y_c = ys_e * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Γ_θy_e = 1/(N_e-1)*θ_c*Y_c'
        Γ_yy_e = 1/(N_e-1)*Y_c*Y_c'
        K = Γ_θy_e * inv(Γ_yy_e + Γ_ϵϵ)

        # Update each ensemble member
        θs_e = θs_e + K*(ys_p-ys_e)
        
    end

    return θs_e

end


"""Runs the batch ensemble randomised maximum likelihood algorithm as presented
in Emerick and Reynolds (2012)."""
function run_batch_enrml(
    f::Function,
    g::Function,
    π::AbstractPrior,
    ys::AbstractVector,
    σ_ϵ::Real,
    β_0::Real,
    N_e::Int;
    verbose::Bool=true
)

    function mean_misfit(ys_lp::AbstractMatrix, ys_p::AbstractMatrix)::Float64
        return Statistics.mean(0.5(y_lp-y_p)'*inv(Γ_ϵ)*(y_lp-y_p) 
            for (y_lp, y_p) ∈ zip(eachcol(ys_lp), eachcol(ys_p)))
    end

    function converged(θs_lp, θs_l, O_lp, O_l, n_it, n_cuts)

        maximum(abs.(θs_lp-θs_l)) < 1e-3 && return true
        abs((O_lp-O_l)/O_l) < 1e-2 && return true
        n_it == 10 && return true
        n_cuts == 5 && return true
        
        return false

    end

    # Sample an ensemble from the prior and run it
    θs_f = reduce(hcat, sample(π, n=N_e))
    ys_f = reduce(hcat, [g(f(θ)) for θ ∈ eachcol(θs_f)])

    # Form the covariance of the errors
    Γ_ϵ = σ_ϵ^2 * Matrix(LinearAlgebra.I, length(ys), length(ys))

    # Calculate the prior covariance of the simulated values
    θ_c = θs_f * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
    Γ_θf = 1/(N_e-1)*θ_c*θ_c'

    # Sample an ensemble of perturbed observations
    ys_p = rand(Distributions.MvNormal(ys, Γ_ϵ), N_e)

    O_l = mean_misfit(ys_f, ys_p)
    
    n_it = 0; n_cuts = 0
    θs_l = copy(θs_f); θs_lp = copy(θs_f); 
    ys_l = copy(ys_f); ys_lp = copy(ys_f); 
    O_lp = O_l
    β_l = β_0

    while true

        # Calculate the ensemble sensitivity
        θ_c = θs_l * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Y_c = ys_l * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        G_l = Y_c*LinearAlgebra.pinv(θ_c)

        while true
            
            # Update the ensemble and run it forward in time
            θs_lp = β_l*θs_f + (1-β_l)*θs_l - 
                β_l*Γ_θf*G_l'*inv(G_l*Γ_θf*G_l'+Γ_ϵ)*(ys_l-ys_p-G_l*(θs_l-θs_f))

            ys_lp = reduce(hcat, [g(f(θ)) for θ ∈ eachcol(θs_lp)])

            O_lp = mean_misfit(ys_lp, ys_p)

            if O_lp < O_l 

                verbose && @info("Step accepted. Increasing step size.")
                β_l = min(2β_l, β_0); n_cuts = 0;
                n_it += 1; break

            else 

                verbose && @info("Step rejected. Decreasing step size.")
                β_l *= 0.5; n_cuts += 1
                n_cuts > 5 && break

            end

        end

        # Check for convergence
        converged(θs_lp, θs_l, O_lp, O_l, n_it, n_cuts) && break
        θs_l = copy(θs_lp); ys_l = copy(ys_lp)
        O_l = O_lp

    end

    return θs_l

end