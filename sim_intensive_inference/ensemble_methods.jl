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
        U_c = vcat(us_e, θs_e) * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Y_c = ys_e * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Γ_uy_e = 1/(N_e-1)*U_c*Y_c'
        Γ_y_e = 1/(N_e-1)*Y_c*Y_c'
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


function run_hi_enkf(
    a::Function,
    b::Function,
    π::AbstractPrior,
    ts::AbstractVector,
    ys::AbstractMatrix,
    σ_ϵ::Real,
    N_e::Int,
    N_u::Int;
    verbose::Bool=true
)

    # Initialise a matrix to store the combined sets of states generated
    us_e_c = Matrix(undef, N_e*N_u, 0)

    # Sample an ensemble of sets of parameters from the prior
    θs_e = reduce(hcat, sample(π, n=N_e))

    for (i, (t, y)) ∈ enumerate(zip(ts, eachcol(ys)))

        # Run the state model (from the beginning) for each ensemble member
        us_e_l = [a(θ, t_1=t) for θ ∈ eachcol(θs_e)]
        
        # Update combined state vectors
        us_e_c = hcat(us_e_c, reduce(vcat, us_e_l)[:, size(us_e_c, 2)+1:end])

        # Form matrices containing the final states and modelled observations
        us_e = reduce(hcat, [u[:, end] for u ∈ us_e_l])
        ys_e = reduce(hcat, [b(θ, u) for (θ, u) ∈ zip(eachcol(θs_e), eachcol(us_e))])

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

        verbose && @info "Iteration $i complete."

    end

    # Run everything to the end with the final set of parameters
    us_e_l = [a(θ) for θ ∈ eachcol(θs_e)]
    us_e_c = hcat(us_e_c, reduce(vcat, us_e_l)[:, size(us_e_c, 2)+1:end])

    return θs_e, us_e_c

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
            Γ_ϵϵ = α * σ_ϵ^2 * Matrix(LinearAlgebra.I, length(y), length(y))
            ys_p = rand(Distributions.MvNormal(y, Γ_ϵϵ), N_e)

            # Compute the Kalman gain
            θ_c = θs_e * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
            Y_c = ys_e * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
            Γ_θy_e = 1/(N_e-1)*θ_c*Y_c'
            Γ_yy_e = 1/(N_e-1)*Y_c*Y_c'
            K = Γ_θy_e * inv(Γ_yy_e + Γ_ϵϵ)

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


function run_es_mda(
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

    for (i, α) ∈ enumerate(αs)

        # Generate the ensemble predictions 
        ys_e = reduce(hcat, [g(f(θ)) for θ ∈ eachcol(θs_e)])

        # Generate a set of perturbed data vectors 
        Γ_ϵ = α * σ_ϵ^2 * Matrix(LinearAlgebra.I, length(ys), length(ys))
        ys_p = rand(Distributions.MvNormal(ys, Γ_ϵ), N_e)

        # Compute the gain
        θ_c = θs_e * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Y_c = ys_e * (LinearAlgebra.I - ones(N_e, N_e)/N_e)
        Γ_θy_e = 1/(N_e-1)*θ_c*Y_c'
        Γ_y_e = 1/(N_e-1)*Y_c*Y_c'
        K = Γ_θy_e * inv(Γ_y_e + Γ_ϵ)

        # Update each ensemble member
        θs_e = θs_e + K*(ys_p-ys_e)

        verbose && @info("Iteration $i complete.")
        
    end

    return θs_e

end


"""Runs the batch ensemble randomised maximum likelihood algorithm as presented
in Emerick and Reynolds (2013)."""
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
    ys_f_l = [f(θ) for θ ∈ eachcol(θs_f)]
    ys_f = reduce(hcat, [g(y) for y ∈ ys_f_l])

    # Plotting.plot_monod_states(
    #     reduce(vcat, ys_f_l),
    #     "Test", "test.pdf"
    # )

    # # Calculate the misfit of each data point 
    # ss = vec(sum((ys_f .- ys).^2, dims=1))
    # println(sort(ss))

    # while Statistics.maximum(ss) > 100
    #     i_max = argmax(ss)
    #     ss = ss[1:N_e .!= i_max]
    #     θs_f = θs_f[:, 1:N_e .!= i_max]
    #     ys_f = ys_f[:, 1:N_e .!= i_max]
    #     N_e -= 1
    # end

    # println(N_e)

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
    ys_l = copy(ys_f); ys_lp = copy(ys_f); ys_lp_l = []
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

            # println(θs_lp)
            ys_lp_l = [f(θ) for θ ∈ eachcol(θs_lp)]
            ys_lp = reduce(hcat, [g(y) for y ∈ ys_lp_l])

            # display(Statistics.maximum(ys_lp, dims=2))
            # display(Statistics.minimum(ys_lp, dims=2))

            O_lp = mean_misfit(ys_lp, ys_p)

            if O_lp < O_l 

                verbose && @info("Step accepted. Increasing step size.")
                β_l = min(2β_l, β_0); n_cuts = 0;
                n_it += 1; break

            else

                verbose && @info("Step rejected. Decreasing step size.")
                β_l *= 0.5; n_cuts += 1
                n_cuts == 5 && break

            end

        end

        # Check for convergence
        converged(θs_lp, θs_l, O_lp, O_l, n_it, n_cuts) && break
        θs_l = copy(θs_lp); ys_l = copy(ys_lp)
        O_l = O_lp

    end

    return θs_l, reduce(vcat, ys_lp_l)

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
    l_max::Int,
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
    θs = [Matrix(undef, N_θ, N_e) for _ ∈ 1:l_max+1]
    ys = [Matrix(undef, N_y, N_e) for _ ∈ 1:l_max+1]
    ys_c = [Matrix(undef, 0, 0) for _ ∈ 1:l_max+1] # TODO: give this the correct dimensions?
    Ss = Vector(undef, l_max+1)
    λs = Vector(undef, l_max+1)

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
    U_θπ, Λ_θπ, _ = LinearAlgebra.svd(Δθ_π) # TODO: truncate somewhere?

    println(Λ_θπ)

    A_π = U_θπ * inv(LinearAlgebra.Diagonal(Λ_θπ))

    θs[1] = copy(θs_π)
    ys_l = [f(θ) for θ ∈ eachcol(θs_π)]
    ys_c[1] = reduce(vcat, ys_l)
    ys[1] = reduce(hcat, [g(y) for y ∈ ys_l])
    Ss[1] = calculate_s(ys[1], ys_p, Γ_ϵ_i)

    l = 2
    λs[l] = 10^floor(log10(Ss[1]/2N_y))

    while l < l_max+1
        
        # Construct matrices of normalised deviations
        Δθ = Γ_sc_isqrt * (θs[l-1] .- Statistics.mean(θs[l-1], dims=2)) / √(N_e-1)
        Δy = Γ_ϵ_isqrt  * (ys[l-1] .- Statistics.mean(ys[l-1], dims=2)) / √(N_e-1)

        U_y, Λ_y, V_y = LinearAlgebra.svd(Δy)

        U_y = U_y[:,1:end-4]
        Λ_y = Λ_y[1:end-4]
        V_y = V_y[:,1:end-4]
        #display(Λ_y)
        Λ_y = LinearAlgebra.Diagonal(Λ_y) # TODO: truncate somewhere?
        #display(inv((λs[l]+1)LinearAlgebra.I * Λ_y^2))

        # Calculate ensemble corrections based on misfit
        δθ_1 = -Γ_sc_sqrt * Δθ * V_y * Λ_y * 
                inv((λs[l]+1)LinearAlgebra.I + Λ_y^2) * 
                U_y' * Γ_ϵ_isqrt * (ys[l-1] - ys_p) 

        # Calculate ensemble corrections based on deviation from prior
        δθ_2 = 0
        if l > 2
            δθ_2 = -Γ_sc_sqrt * Δθ * V_y * 
                    inv((λs[l]+1)LinearAlgebra.I + Λ_y^2) * 
                    V_y' * Δθ' * A_π * A_π' * 
                    Γ_sc_sqrt * (θs[l-1] - θs_π)
        end

        θs[l] = θs[l-1] + δθ_1 .+ δθ_2
        ys_l = [f(θ) for θ ∈ eachcol(θs[l])]
        ys_c[l] = reduce(vcat, ys_l)
        ys[l] = reduce(hcat, [g(y) for y ∈ ys_l])
        Ss[l] = calculate_s(ys[l], ys_p, Γ_ϵ_i)

        if Ss[l] ≤ Ss[l-1]
            if 1-Ss[l]/Ss[l-1] ≤ C1 || LinearAlgebra.norm(θs[l] - θs[l-1]) ≤ C2
                return θs[1:l], ys_c[1:l], Ss[1:l], λs[1:l]
            else
                l += 1
                λs[l] = max(λs[l-1]/γ, λ_min)
                verbose && @info "Step accepted: reduced λ to $(λs[l])."
            end 
        else 
            λs[l] *= γ
            verbose && @info "Step rejected: increased λ to $(λs[l])."
        end

    end

    return θs[1:l-1], ys_c[1:l-1], Ss[1:l-1], λs[1:l-1]

end