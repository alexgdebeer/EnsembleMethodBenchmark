"""Runs the EnKF algorithm, with the parameters augmented to the states."""
function run_enkf(
    a::Function,
    b::Function,
    π::AbstractPrior,
    u_0::AbstractVector,
    ts::AbstractVector,
    ys::AbstractMatrix,
    σ_ϵ::Real,
    N_e::Int;
    verbose::Bool=true
)

    prepend!(ts, 0.0)

    N_u = length(u_0)

    # Initialise a matrix to store the combined sets of states generated
    us_e_c = Matrix(undef, N_e*N_u, 0)

    # Generate an initial ensemble from the prior
    θs_e = reduce(hcat, sample(π, n=N_e))
    us_e = repeat(u_0', N_e)'

    for (i, (t_0, t_1, y)) ∈ enumerate(zip(ts[1:(end-1)], ts[2:end], eachcol(ys)))

        # Run each ensemble member forward to the current observation time 
        us_e_l = [a(θ, u_0=u, t_0=t_0, t_1=t_1) 
            for (θ, u) ∈ zip(eachcol(θs_e), eachcol(us_e))]

        # Update combined state vectors
        us_e_c = hcat(us_e_c[:, 1:end-1], reduce(vcat, us_e_l))

        # Form matrices containing the final states and modelled observations
        us_e = reduce(hcat, [u[:, end] for u ∈ us_e_l])
        ys_e = reduce(hcat, [b(θ, u) for (θ, u) ∈ zip(eachcol(θs_e), eachcol(us_e))])

        # Generate a set of perturbed data vectors 
        Γ_ϵ = σ_ϵ^2 * Matrix(LinearAlgebra.I, length(y), length(y))
        ys_p = rand(Distributions.MvNormal(y, Γ_ϵ), N_e)

        # Compute the Kalman gain
        Δu = vcat(us_e, θs_e) .- Statistics.mean(vcat(us_e, θs_e), dims=2)
        Δy = ys_e .- Statistics.mean(ys_e, dims=2)
        Γ_uy_e = 1/(N_e-1)*Δu*Δy'
        Γ_y_e = 1/(N_e-1)*Δy*Δy'
        K = Γ_uy_e * inv(Γ_y_e + Γ_ϵ)
        
        # Update the ensemble
        uθs_e = vcat(us_e, θs_e) + K*(ys_p-ys_e)
        us_e, θs_e = uθs_e[1:N_u, :], uθs_e[(N_u+1):end, :]

        verbose && @info("Iteration $i complete.")

    end

    # Run each ensemble member to the end of the time period of interest 
    us_e_l = [a(θ, u_0=u, t_0=ts[end]) 
        for (θ, u) ∈ zip(eachcol(θs_e), eachcol(us_e))]

    # Update combined state vectors
    us_e_c = hcat(us_e_c[:, 1:end-1], reduce(vcat, us_e_l))

    return θs_e, us_e_c

end


"""Runs the half-iteration EnKF algorithm."""
function run_hi_enkf(
    a::Function,
    b::Function,
    π::Distribution,
    ts::AbstractVector,
    ys_obs::AbstractMatrix,
    σ_ϵ::Real,
    N_e::Int,
    N_u::Int;
    verbose::Bool=true
)

    N_i = length(ts)
    N_θ = length(π.μ)
    N_y = size(ys_obs, 1)

    us = zeros(N_u, N_e, N_i+1)
    θs = zeros(N_θ, N_e, N_i+1)
    ys = zeros(N_y, N_e, N_i+1)

    θs[:,:,1] = rand(π, N_e)

    for (i, t, y) ∈ zip(2:N_i+1, ts, eachcol(ys_obs))

        # Generate the states and observations at the current time of interest
        us[:,:,i-1] = reduce(hcat, [a(θs[:,j,i-1], t_1=t)[:,end] for j ∈ 1:N_e])
        ys[:,:,i-1] = reduce(hcat, [b(θs[:,j,i-1], us[:,j,i-1]) for j ∈ 1:N_e])

        # Generate a set of perturbations
        ϵs = rand(MvNormal(zeros(length(y)), σ_ϵ^2 * I), N_e)

        # Update each ensemble member
        K = kalman_gain(θs[:,:,i-1], ys[:,:,i-1], σ_ϵ^2 * I)
        θs[:,:,i] = θs[:,:,i-1] + K * (y .+ ϵs .- ys[:,:,i-1])

        verbose && @info "Iteration $(i-1) complete."

    end

    us[:,:,end] = reduce(hcat, [a(θs[:,j,end], t_1=ts[end])[:,end] for j ∈ 1:N_e])
    ys[:,:,end] = reduce(hcat, [b(θs[:,j,end], us[:,j,end]) for j ∈ 1:N_e])

    return θs, us, ys

end


function run_hi_enkf_mda(
    H::Function,
    π::AbstractPrior,
    ts::AbstractVector,
    ys::AbstractMatrix,
    σ_ϵ::Real,
    αs::AbstractVector,
    N_e::Int;
    verbose::Bool=true
)

    # Sample an ensemble of sets of parameters from the prior
    θs_e = reduce(hcat, sample(π, n=N_e))

    for (i, (t, y)) ∈ enumerate(zip(ts, eachcol(ys)))

        for α ∈ αs

            # Generate the ensemble predictions for the current time
            ys_e = reduce(hcat, [H(θ, t) for θ ∈ eachcol(θs_e)])

            # Generate a set of perturbed data vectors 
            Γ_ϵ = α * σ_ϵ^2 * Matrix(LinearAlgebra.I, length(y), length(y))
            ys_p = rand(Distributions.MvNormal(y, Γ_ϵ), N_e)

            # Compute the Kalman gain
            Δθ = θs_e .- Statistics.mean(θs_e, dims=2)
            Δy = ys_e .- Statistics.mean(ys_e, dims=2)
            Γ_θy_e = 1/(N_e-1)*Δθ*Δy'
            Γ_y_e = 1/(N_e-1)*Δy*Δy'
            K = Γ_θy_e * inv(Γ_y_e + Γ_ϵ)

            # Update each ensemble member
            θs_e = θs_e + K*(ys_p-ys_e)

        end

        verbose && @info "Iteration $i complete."

    end

    return θs_e

end


function run_es(
    f::Function,
    g::Function,
    π::Distribution,
    L::Distribution,
    N_e::Int;
    verbose::Bool=true
)

    N_θ = length(π.μ)
    N_y = length(L.μ)

    θs = zeros(N_θ, N_e, 2)
    ys = zeros(N_y, N_e, 2)

    # Sample an initial ensemble and predictions
    θs[:,:,1] = rand(π, N_e)
    ys[:,:,1] = reduce(hcat, [g(f(θ)) for θ ∈ eachcol(θs[:,:,1])])

    # Generate a set of perturbations
    ϵs = rand(Distributions.MvNormal(zeros(N_θ), Γ_ϵ), N_e)

    # Form the updated ensemble and generate the updated ensemble predictions
    K = kalman_gain(θs[:,:,1], ys[:,:,1], L.Σ)
    θs[:,:,2] = θs[:,:,1] + K * (L.μ .+ ϵs .- ys[:,:,1])
    ys[:,:,2] = reduce(vcat, [f(θ) for θ ∈ eachcol(θs[:,:,2])])

    return θs, ys

end


function run_es_mda(
    f::Function,
    g::Function,
    π::Distribution,
    L::Distribution,
    αs::AbstractVector,
    N_e::Int;
    verbose::Bool=true
)

    sum(1 ./ αs) ≉ 1.0 && error("Reciprocals of α values do not sum to 1.")

    N_i = length(αs)
    N_θ = length(π.μ)
    N_y = length(L.μ)

    θs = zeros(N_θ, N_e, N_i+1)
    ys = zeros(N_y, N_e, N_i+1)

    # Sample an initial ensemble and predictions
    θs[:,:,1] = rand(π, N_e)
    ys[:,:,1] = reduce(hcat, [g(f(θ)) for θ ∈ eachcol(θs[:,:,1])])

    for i ∈ 2:N_i+1

        # Generate a set of perturbations
        ϵs = rand(MvNormal(zeros(N_θ), αs[i-1]*L.Σ), N_e)

        # Generate the new ensemble and associated predictions
        K = kalman_gain(θs[:,:,i-1], ys[:,:,i-1], αs[i-1]*L.Σ)
        θs[:,:,i] = θs[:,:,i-1] + K * (L.μ .+ ϵs .- ys[:,:,i-1])
        ys[:,:,i] = reduce(hcat, [g(f(θ)) for θ ∈ eachcol(θs[:,:,i])])

        verbose && @info "Iteration $(i-1) complete."
        
    end

    return θs, ys

end


"""Runs the batch ensemble randomised maximum likelihood algorithm as presented
in Emerick and Reynolds (2013)."""
function run_batch_enrml(
    f::Function,
    g::Function,
    π::AbstractPrior,
    ys_obs::AbstractVector,
    σ_ϵ::Real,
    β_0::Real,
    i_max::Int,
    N_e::Int;
    verbose::Bool=true
)

    function mean_misfit(ys::AbstractMatrix, ys_p::AbstractMatrix)::Float64
        return Statistics.mean(0.5(y-y_p)'*inv(Γ_ϵ)*(y-y_p) 
            for (y, y_p) ∈ zip(eachcol(ys), eachcol(ys_p)))
    end

    function converged(θs, θs_p, S, S_p, i, n_cuts)

        maximum(abs.(θs-θs_p)) < 1e-3 && return true
        abs((S-S_p)/S_p) < 1e-2 && return true
        i == i_max+1 && return true
        n_cuts == 5 && return true

        return false

    end

    N_θ = length(π.μ)
    N_y = length(ys_obs)

    # Initialise some vectors
    θs = [Matrix(undef, N_θ, N_e) for _ ∈ 1:i_max+1]
    ys = [Matrix(undef, N_y, N_e) for _ ∈ 1:i_max+1]
    ys_c = [Matrix(undef, 0, 0) for _ ∈ 1:i_max+1] # TODO: give this the correct dimensions?
    Ss = Vector(undef, i_max+1)
    βs = Vector(undef, i_max+1)

    # Sample an ensemble from the prior and run it forward in time
    θs[1] = reduce(hcat, sample(π, n=N_e))
    ys_l = [f(θ) for θ ∈ eachcol(θs[1])]
    ys_c[1] = reduce(vcat, ys_l)
    ys[1] = reduce(hcat, [g(y) for y ∈ ys_l])

    # Extract the covariances of the parameters and errors
    Γ_θ = π.Σ
    Γ_ϵ = σ_ϵ^2 * Matrix(LinearAlgebra.I, N_y, N_y)

    # Sample an ensemble of perturbed observations
    ys_p = rand(Distributions.MvNormal(ys_obs, Γ_ϵ), N_e)
    
    Ss[1] = mean_misfit(ys[1], ys_p)
    βs[1] = β_0

    n_cuts = 0
    i = 1

    while true

        # Calculate the ensemble sensitivity
        Δθ = θs[i] .- Statistics.mean(θs[i], dims=2)
        Δy = ys[i] .- Statistics.mean(ys[i], dims=2)
        G = Δy * LinearAlgebra.pinv(Δθ)

        while true

            # Update the ensemble and run it forward in time
            θs[i+1] = βs[i]*θs[1] + (1-βs[i])*θs[i] - 
                βs[i]*Γ_θ*G'*inv(G*Γ_θ*G'+Γ_ϵ)*(ys[i]-ys_p-G*(θs[i]-θs[1]))
            
            ys_l = [f(θ) for θ ∈ eachcol(θs[i+1])]
            ys_c[i+1] = reduce(vcat, ys_l)
            ys[i+1] = reduce(hcat, [g(y) for y ∈ ys_l])

            Ss[i+1] = mean_misfit(ys[i+1], ys_p)

            if Ss[i+1] < Ss[i] 

                βs[i+1] = min(2βs[i], β_0); n_cuts = 0;
                verbose && @info "Step accepted. Step size: $(βs[i+1])."
                
                if converged(θs[i+1], θs[i], Ss[i+1], Ss[i], i, n_cuts) 
                    return θs[1:i+1], ys_c[1:i+1], Ss[1:i+1], βs[1:i+1]
                end

                i += 1; break

            else

                n_cuts += 1
                βs[i] *= 0.5
                verbose && @info "Step rejected. Step size: $(βs[i])."

                if n_cuts == 5 
                    return θs[1:i], ys_c[1:i], Ss[1:i], βs[1:i]
                end

            end

        end

    end

end


"""Runs the Levenberg-Marquardt iterative ensemble smoother, as described in 
Chen (2013)."""
function run_lm_enrml(
    f::Function, 
    g::Function,
    π::AbstractPrior,
    ys_obs::AbstractVector, 
    σ_ϵ::Real, 
    γ::Real,
    i_max::Int,
    N_e::Int;
    λ_min::Real=0.01,
    verbose::Bool=true
)

    # TODO: define these properly 
    C1 = 0
    C2 = 0

    """TODO: update to return standard deviation of S."""
    function calculate_s(ys, ys_p, Γ_ϵ_i)
        return Statistics.mean([(y-y_p)' * Γ_ϵ_i * (y-y_p) for (y, y_p) ∈ zip(eachcol(ys), eachcol(ys_p))])
    end

    N_θ = length(π.μ)
    N_y = length(ys_obs)

    # Initialise some vectors
    θs = [Matrix(undef, N_θ, N_e) for _ ∈ 1:i_max+1]
    ys = [Matrix(undef, N_y, N_e) for _ ∈ 1:i_max+1]
    ys_c = [Matrix(undef, 0, 0) for _ ∈ 1:i_max+1] # TODO: give this the correct dimensions?
    Ss = Vector(undef, i_max+1)
    λs = Vector(undef, i_max+1)

    # Generate the covariance of the observations
    Γ_ϵ = σ_ϵ^2 * Matrix(LinearAlgebra.I, length(ys_obs), length(ys_obs))
    Γ_ϵ_i = inv(Γ_ϵ)
    Γ_ϵ_isqrt = sqrt(Γ_ϵ_i)

    # Calculate the prior scaling matrix, the scaled prior deviations, and the 
    # SVD of the scaled deviations
    Γ_sc = LinearAlgebra.Diagonal(sqrt.(LinearAlgebra.diag(π.Σ)))
    Γ_sc_sqrt = sqrt(Γ_sc)
    Γ_sc_isqrt = inv(Γ_sc_sqrt)

    # Sample a set of parameters from the prior
    θs_π = reduce(hcat, sample(π, n=N_e))

    # Generate some pertubed observations
    ys_p = rand(Distributions.MvNormal(ys_obs, Γ_ϵ), N_e)

    Δθ_π = Γ_sc_isqrt * (θs_π .- Statistics.mean(θs_π, dims=2)) / sqrt(N_e-1)
    U_θπ, Λ_θπ, _ = tsvd(Δθ_π)

    A_π = U_θπ * inv(LinearAlgebra.Diagonal(Λ_θπ))

    θs[1] = copy(θs_π)
    ys_l = [f(θ) for θ ∈ eachcol(θs_π)]
    ys_c[1] = reduce(vcat, ys_l)
    ys[1] = reduce(hcat, [g(y) for y ∈ ys_l])
    Ss[1] = calculate_s(ys[1], ys_p, Γ_ϵ_i)

    i = 2
    λs[i] = 10000 # 10^floor(log10(Ss[1]/2N_y))

    while i < i_max+1
        
        # Construct matrices of normalised deviations
        Δθ = Γ_sc_isqrt * (θs[i-1] .- Statistics.mean(θs[i-1], dims=2)) / √(N_e-1)
        Δy = Γ_ϵ_isqrt  * (ys[i-1] .- Statistics.mean(ys[i-1], dims=2)) / √(N_e-1)

        U_y, Λ_y, V_y = tsvd(Δy)
        Λ_y = LinearAlgebra.Diagonal(Λ_y)

        # Calculate ensemble corrections based on misfit
        δθ_1 = -Γ_sc_sqrt * Δθ * V_y * Λ_y * 
                inv((λs[i]+1)LinearAlgebra.I + Λ_y^2) * 
                U_y' * Γ_ϵ_isqrt * (ys[i-1] - ys_p) 

        # Calculate ensemble corrections based on deviation from prior
        δθ_2 = -Γ_sc_sqrt * Δθ * V_y * 
                inv((λs[i]+1)LinearAlgebra.I + Λ_y^2) * 
                V_y' * Δθ' * A_π * A_π' * 
                Γ_sc_sqrt * (θs[i-1] - θs_π)

        θs[i] = θs[i-1] + δθ_1 + δθ_2
        ys_l = [f(θ) for θ ∈ eachcol(θs[i])]
        ys_c[i] = reduce(vcat, ys_l)
        ys[i] = reduce(hcat, [g(y) for y ∈ ys_l])
        Ss[i] = calculate_s(ys[i], ys_p, Γ_ϵ_i)

        if Ss[i] ≤ Ss[i-1]
            if 1-Ss[i]/Ss[i-1] ≤ C1 || LinearAlgebra.norm(θs[i] - θs[i-1]) ≤ C2
                return θs[1:i], ys_c[1:i], Ss[1:i], λs[1:i]
            else
                i += 1
                λs[i] = max(λs[i-1]/γ, λ_min)
                verbose && @info "Step accepted: λ is now $(λs[i])."
            end 
        else 
            λs[i] *= γ
            verbose && @info "Step rejected: λ is now $(λs[i])."
        end

    end

    return θs[1:i-1], ys_c[1:i-1], Ss[1:i-1], λs[1:i-1]

end


"""Runs the weighted ES-MDA algorithm, as described in Stordal (2015)."""
function run_wes_mda(
    f::Function,
    g::Function,
    π::Distribution,
    L::Distribution,
    αs::AbstractVector,
    N_e::Int;
    min_ess::Real=N_e/2,
    verbose::Bool=true
)

    βs = cumsum(1 ./ αs)
    βs[end] ≉ 1.0 && error("Reciprocals of α values do not sum to 1.")

    N_i = length(αs)
    N_θ = length(π.μ)
    N_y = length(L.μ)

    θs = zeros(N_θ, N_e, N_i+1)
    ys = zeros(N_y, N_e, N_i+1)
    ws = zeros(N_i+1, N_e)

    # Sample an initial ensemble and predictions
    θs[:,:,1] = rand(π, N_e)
    ys[:,:,1] = reduce(hcat, [g(f(θ)) for θ ∈ eachcol(θs[:,:,1])])
    ws[1,:] = ones(N_e)/N_e; logws = zeros(N_e)

    for i ∈ 2:N_i+1

        # Generate a set of perturbations
        ϵs = rand(MvNormal(zeros(N_y), αs[i-1]*L.Σ), N_e)

        # Generate the new ensemble and associated predictions
        K = kalman_gain(θs[:,:,i-1], ys[:,:,i-1], αs[i-1]*L.Σ)
        θs[:,:,i] = θs[:,:,i-1] + K * (L.μ .+ ϵs .- ys[:,:,i-1])
        ys[:,:,i] = reduce(hcat, [g(f(θ)) for θ ∈ eachcol(θs[:,:,i])])

        # Generate forward and backwards kernels
        μ_Ks = θs[:,:,i-1] + K * (L.μ .- ys[:,:,i-1])
        Γ_K = Hermitian(K * αs[i-1]*L.Σ * K' + 1e-8I)
        K = MvNormal(zeros(N_θ), Γ_K)
        T = MvNormal(zeros(N_θ), Γ_K)

        # Calculate the weights
        β_p = i ≤ 2 ? 0.0 : βs[i-2]
        logws .+= [
            logpdf(π, θs[:,j,i]) + βs[i-1] * logpdf(L, ys[:,j,i]) - 
            logpdf(π, θs[:,j,i-1]) - β_p * logpdf(L, ys[:,j,i-1]) + 
            logpdf(T, θs[:,j,i-1]-θs[:,j,i]) -
            logpdf(K, θs[:,j,i]-μ_Ks[:,j])
                for j ∈ 1:N_e]

        # Re-scale, exponentiate and normalise weights
        logws = rescale_logws(logws)
        ws[i,:] = normalise_ws(exp.(logws))

        if i != N_i+1 && ess(ws[i,:]) < min_ess

            @info "ESS below acceptable threshold ($(ess(ws[i,:]))). Resampling."
            
            # Resample
            is = resample_ws(ws[i,:])
            θs[:,:,i] = θs[:,is,i]
            ys[:,:,i] = ys[:,is,i]
            
            # Reset the weights
            ws[i,:] = ones(N_e)/N_e
            logws = zeros(N_e)

        end

        verbose && @info "Iteration $(i-1) complete."
        
    end

    # TODO: return full set of ys?
    return θs, ys, ws
    
end


"""Runs the weighted ES-MDA algorithm (my verison)."""
function run_wes_mda_alt(
    f::Function,
    g::Function,
    π::Distribution,
    L::Distribution,
    αs::AbstractVector,
    N_e::Int;
    min_ess::Real=N_e/2,
    verbose::Bool=true
)

    βs = cumsum(1 ./ αs)
    βs[end] ≉ 1.0 && error("Reciprocals of α values do not sum to 1.")

    N_i = length(αs)
    N_θ = length(π.μ)
    N_y = length(L.μ)

    θs = zeros(N_θ, N_e, N_i+1)
    ys = zeros(N_y, N_e, N_i+1)
    ws = zeros(N_i+1, N_e)

    # Sample an initial ensemble and predictions
    θs[:,:,1] = rand(π, N_e)
    ys[:,:,1] = reduce(hcat, [g(f(θ)) for θ ∈ eachcol(θs[:,:,1])])
    ws[1,:] = ones(N_e)/N_e; logws = zeros(N_e)

    for i ∈ 2:N_i+1

        # Generate a set of perturbations
        ϵs = rand(MvNormal(zeros(N_y), αs[i-1]*L.Σ), N_e)

        # Generate the new set of particles and associated predictions
        K = kalman_gain(θs[:,:,i-1], ys[:,:,i-1], αs[i-1]*L.Σ)
        θs[:,:,i] = θs[:,:,i-1] + K * (L.μ .+ ϵs .- ys[:,:,i-1])
        ys[:,:,i] = reduce(hcat, [g(f(θ)) for θ ∈ eachcol(θs[:,:,i])])

        # Generate the importance distribution
        μ_Ks = θs[:,:,i-1] + K * (L.μ .- ys[:,:,i-1])
        Γ_K = Hermitian(K * αs[i-1]*L.Σ * K' + 1e-8I)
        K = MixtureModel(MvNormal, [(μ, Γ_K) for μ ∈ eachcol(μ_Ks)], ws[i-1,:])

        # Increment the weights
        logws .+= [
            logpdf(π, θs[:,j,i]) + 
            βs[i-1] * logpdf(L, ys[:,j,i]) - 
            logpdf(K, θs[:,j,i]) 
                for j ∈ 1:N_e]

        # Re-scale, exponentiate and normalise weights
        logws = rescale_logws(logws)
        ws[i,:] = normalise_ws(exp.(logws))

        if i != N_i+1 && ess(ws[i,:]) < min_ess

            @info "ESS below acceptable threshold ($(ess(ws[i,:]))). Resampling."
            
            # Resample
            is = resample_ws(ws[i,:])
            θs[:,:,i] = θs[:,is,i]
            ys[:,:,i] = ys[:,is,i]
            
            # Reset the weights
            ws[i,:] = ones(N_e)/N_e
            logws = zeros(N_e)

        end

        verbose && @info "Iteration $(i-1) complete."
        
    end

    return θs, ys, ws
    
end