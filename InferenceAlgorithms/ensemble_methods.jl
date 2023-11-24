function compute_enrml_gain(
    Δη::AbstractMatrix,
    UG::AbstractMatrix,
    ΛG::AbstractMatrix,
    VG::AbstractMatrix,
    Γ_e_sqi::AbstractMatrix,
    λ::Real, 
)
    return Δη * VG * ΛG * inv((λ+1)I + ΛG.^2) * UG' * Γ_e_sqi
end

function compute_Δs(
    xs::AbstractMatrix, 
    N::Int
)::AbstractMatrix

    return (xs .- mean(xs, dims=2)) ./ √(N-1)

end

function tsvd(
    A::AbstractMatrix; 
    e::Real=0.99
)::Tuple{AbstractMatrix, AbstractVector, AbstractMatrix}

    U, Λ, V = svd(A)
    minimum(Λ) < 0 && error(minimum(Λ))
    λ_cum = cumsum(Λ)

    for i ∈ 1:length(Λ)
        if λ_cum[i] / λ_cum[end] ≥ e 
            return U[:, 1:i], Λ[1:i], V[:, 1:i]
        end
    end

end

function run_enrml(
    F::Function, 
    G::Function,
    pr::MaternField,
    d_obs::AbstractVector,
    μ_e::AbstractVector,
    Γ_e::AbstractMatrix,
    Ne::Int,
    NF::Int,
    i_max::Int;
    γ::Real=10,
    λ_min::Real=0.01,
    localisation::Bool=false, # TODO: add localisation options
    verbose::Bool=true
)

    # TODO: make these functions available to both algorithms
    compute_θs(ηs) = hcat([transform(pr, η_i) for η_i ∈ eachcol(ηs)]...)
    compute_Fs(θs) = hcat([F(θ_i) for θ_i ∈ eachcol(θs)]...)
    compute_Gs(Fs) = hcat([G(F_i) for F_i ∈ eachcol(Fs)]...)

    """Returns the mean and standard deviations of the sum-of-squares data 
    misfit of each ensemble member."""
    # TODO: this can probably be combined with the equivalent function in EKI-DMC
    function compute_S(Gs, ds_p, Γ_e_sqi)
        ΔG_sums = sum((Γ_e_sqi * (Gs - ds_p)).^2, dims=1)
        return mean(ΔG_sums)
    end

    # Define convergence parameters
    ΔS_min = 0.01 # Minimum reduction in objective
    Δη_min = 0.5 # Greatest allowable change in a parameter of an ensemble member (prior standard deviations)

    NG = length(d_obs)

    ηs = zeros(pr.Nη, Ne, i_max+1)
    θs = zeros(pr.Nθ, Ne, i_max+1)
    Fs = zeros(NF, Ne, i_max+1)
    Gs = zeros(NG, Ne, i_max+1)
    Ss = zeros(i_max+1)
    λs = zeros(i_max+1)

    # Define set of indices of successful parameter sets
    inds = 1:Ne

    # Compute scaling matrices # TODO: edit
    Γ_e_sqi = sqrt(inv(Γ_e))

    ds_p = rand(MvNormal(d_obs, Γ_e), Ne)

    ηs[:, :, 1] = rand(pr, Ne)
    θs[:, :, 1] = compute_θs(ηs[:, :, 1])
    Fs[:, :, 1] = compute_Fs(θs[:, :, 1])
    Gs[:, :, 1] = compute_Gs(Fs[:, :, 1])

    Ss[1] = compute_S(Gs[:, :, 1], ds_p, Γ_e_sqi)
    λs[1] = 10^floor(log10(Ss[1] / 2NG))

    Δη_pri = compute_Δs(ηs[:, :, 1], length(inds))
    Uη_pri, Λη_pri, = tsvd(Δη_pri)

    if verbose
        println("It. | status   | ΔS       | Δη_max   | λ")
    end

    i = 1
    n_cuts = 0

    while i ≤ i_max

        Δη = compute_Δs(ηs[:, :, i], Ne)
        ΔG = compute_Δs(Gs[:, :, i], Ne)

        UG, ΛG, VG = tsvd(Γ_e_sqi * ΔG)
        K = compute_enrml_gain(Δη, UG, Diagonal(ΛG), VG, Γ_e_sqi, λs[i])

        # Calculate corrections based on prior deviations
        δη_pri = Δη * VG * inv((λs[i] + 1)I + Diagonal(ΛG).^2) * VG' * Δη' *
                 Uη_pri * Diagonal(1 ./ Λη_pri.^2) * Uη_pri' *
                 (ηs[:, :, i] - ηs[:, :, 1])

        # Calculate corrections based on fit to observations
        δη_obs = K * (Gs[:,:,i] .+ μ_e .- ds_p)

        ηs[:, :, i+1] = ηs[:, :, i] - δη_pri - δη_obs
        θs[:, :, i+1] = compute_θs(ηs[:, :, i+1])
        Fs[:, :, i+1] = compute_Fs(θs[:, :, i+1])
        Gs[:, :, i+1] = compute_Gs(Fs[:, :, i+1])

        Ss[i+1] = compute_S(Gs[:, :, i+1], ds_p, Γ_e_sqi)

        if Ss[i+1] ≤ Ss[i]

            n_cuts = 0
            ΔS = 1 - Ss[i+1] / Ss[i]
            Δη_max = maximum(abs.(ηs[:, :, i+1] - ηs[:, :, i]))

            # Check for convergence
            if (ΔS ≤ ΔS_min) && (Δη_max ≤ Δη_min) # TODO: change S to Φ potentially?
                @info "Convergence criteria met."
                return ηs[:, :, 1:i+1], θs[:, :, 1:i+1], Fs[:, :, 1:i+1], Gs[:, :, 1:i+1], Ss[1:i+1], λs[1:i]
            end
            
            i += 1
            λs[i] = max(λs[i-1] / γ, λ_min)

            if verbose
                @printf(
                    "%2i  | accepted | %.2e | %.2e | %.2e \n",
                    i-1, ΔS, Δη_max, λs[i]
                )
            end
             
        else 

            n_cuts += 1
            if n_cuts == 5 
                @info "Terminating algorithm: $n_cuts consecutive cuts to the step size."
                return ηs[:, :, 1:i], θs[:, :, 1:i], Fs[:, :, 1:i], Gs[:, :, 1:i], Ss[1:i], λs[1:i]
            end

            λs[i] *= γ
            
            if verbose
                @printf(
                    "%2i  | rejected | -------- | -------- | %.2e \n",
                    i-1, λs[i]
                )
            end

        end

    end

    return ηs[:, :, 1:i], θs[:, :, 1:i], Fs[:, :, 1:i], Gs[:, :, 1:i], Ss[1:i], λs[1:i-1]

end

"""Runs an ensemble of sets of parameters, and returns the results."""
function run_ensemble(
    ηs::AbstractMatrix, 
    F::Function, 
    G::Function, 
    pr::MaternField
)

    θs = hcat([transform(pr, η_i) for η_i ∈ eachcol(ηs)]...)
    Fs = hcat([F(θ_i) for θ_i ∈ eachcol(θs)]...)
    Gs = hcat([G(F_i) for F_i ∈ eachcol(Fs)]...)

    return θs, Fs, Gs

end

"""Uses the standard EKI update to update the current ensemble."""
function eki_update(
    ηs_i::AbstractMatrix, 
    Gs_i::AbstractMatrix, 
    α_i::Real, 
    y::AbstractVector, 
    μ_e::AbstractVector, 
    C_e::AbstractMatrix
)

    

    # TODO: tidy this up... (could use a block approach, similar to EnsembleKalmanProcesses)
    Ne = size(ηs_i, 2)
    Δη = compute_Δs(ηs_i, Ne)
    ΔG = compute_Δs(Gs_i, Ne)

    C_GG = ΔG * ΔG'
    C_ηG = Δη * ΔG'

    # TODO: covariance localisation

    ys = rand(MvNormal(y, α_i * C_e), Ne)

    return ηs_i + C_ηG * inv(C_GG + α_i*C_e) * (ys - Gs_i .- μ_e)

end

"""Computes the EKI inflation factor following Iglesias and Yang 
(2021)."""
function compute_α_dmc(
    t::Real, 
    Gs::AbstractMatrix, 
    μ_e::AbstractVector,
    y::AbstractVector, 
    NG::Int, 
    C_e_invsqrt::AbstractMatrix
)

    φs = 0.5 * sum((C_e_invsqrt * (y .- Gs .- μ_e)).^2, dims=1)

    μ_φ = mean(φs)
    var_φ = var(φs)
    
    α_inv = max(NG / 2μ_φ, √(NG / 2var_φ))
    α_inv = min(α_inv, 1-t)

    return α_inv ^ -1

end

"""Runs the EKI-DMC algorithm described in Iglesias and Yang (2021)."""
function run_eki_dmc(
    F::Function,
    G::Function,
    pr::MaternField, 
    y::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    Ne::Int,
    localisation::Bool=false, # TODO: add localisation options
    verbose::Bool=true
)

    C_e_invsqrt = √(inv(C_e))

    NG = length(y)

    ηs_i = rand(pr, Ne)
    θs_i, Fs_i, Gs_i = run_ensemble(ηs_i, F, G, pr)

    ηs = [ηs_i]
    θs = [θs_i]
    Fs = [Fs_i]
    Gs = [Gs_i]

    i = 0
    t = 0
    converged = false

    while !converged

        α_i = compute_α_dmc(t, Gs_i, μ_e, y, NG, C_e_invsqrt)
        t += α_i^-1
        if abs(t - 1.0) < 1e-8
            converged = true
        end 

        ηs_i = eki_update(ηs_i, Gs_i, α_i, y, μ_e, C_e)
        θs_i, Fs_i, Gs_i = run_ensemble(ηs_i, F, G, pr)

        push!(ηs, ηs_i)
        push!(θs, θs_i)
        push!(Fs, Fs_i)
        push!(Gs, Gs_i)

        i += 1
        if verbose
            @info "Iteration $i complete. t = $t." # TODO: improve this printing...
        end

    end

    # TODO: make into function?
    ηs = cat(ηs..., dims=3)
    θs = cat(θs..., dims=3)
    Fs = cat(Fs..., dims=3)
    Gs = cat(Gs..., dims=3)

    return ηs, θs, Fs, Gs

end

# TODO: use these in EKI and EnRML
compute_θs(ηs, pr) = hcat([transform(pr, η_i) for η_i ∈ eachcol(ηs)]...)
compute_Fs(θs, F) = hcat([F(θ_i) for θ_i ∈ eachcol(θs)]...)
compute_Gs(Fs, G) = hcat([G(F_i) for F_i ∈ eachcol(Fs)]...)

# TODO: read ALDI paper and do the generalised square root 
function run_eks(
    F::Function,
    G::Function,
    pr::MaternField,
    d_obs::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    Ne::Int;
    Δt₀::Real=10.0,
    t_stop::Real=2.0,
    verbose::Bool=true
)

    NG = length(d_obs)

    ηs = [rand(pr, Ne)]
    θs = [compute_θs(ηs[1], pr)]
    Fs = [compute_Fs(θs[1], F)]
    Gs = [compute_Gs(Fs[1], G)]

    verbose && println("It. | Δt       | t        | misfit ")

    t = 0
    i = 1
    while true

        Ḡ = mean(Gs[end], dims=2)
        η̄ = mean(ηs[end], dims=2)
       
        C_ηη = cov(ηs[end], dims=2, corrected=false)
        D = (1.0 / Ne) * (Gs[end] .- Ḡ)' * (C_e \ (Gs[end] .+ μ_e .- d_obs))
        ζ = rand(MvNormal(C_ηη + 1e-8 * I), Ne)
        
        Δt = Δt₀ / (norm(D) + 1e-6)
        t += Δt

        μ_misfit = mean(abs.(Gs[end] .+ μ_e .- d_obs))
        verbose && @printf "%3i | %.2e | %.2e | %.2e \n" i Δt t μ_misfit
        
        A_n = I + Δt * C_ηη
        B_n = ηs[end] - 
            Δt * (ηs[end] .- η̄) * D +
            Δt * ((NG + 1) / Ne) * (ηs[end] .- η̄)

        ηs_i = (A_n \ B_n) + √(2 * Δt) * ζ
    
        push!(ηs, ηs_i)
        push!(θs, compute_θs(ηs[end], pr))
        push!(Fs, compute_Fs(θs[end], F))
        push!(Gs, compute_Gs(Fs[end], G))

        i += 1
        t ≥ t_stop && break

    end

    return ηs, θs, Fs, Gs

end