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

function run_eki_dmc(
    F::Function,
    G::Function,
    pr::MaternField, 
    d_obs::AbstractVector,
    μ_e::AbstractVector,
    Γ_e::AbstractMatrix,
    L_e::AbstractMatrix,
    Ne::Int,
    localisation::Bool=false, # TODO: add localisation options
    verbose::Bool=true
)

    compute_θs(ηs) = hcat([transform(pr, η_i) for η_i ∈ eachcol(ηs)]...)
    compute_Fs(θs) = hcat([F(θ_i) for θ_i ∈ eachcol(θs)]...)
    compute_Gs(Fs) = hcat([G(F_i) for F_i ∈ eachcol(Fs)]...)

    NG = length(d_obs)

    ηs = [rand(pr, Ne)]
    θs = [compute_θs(ηs[1])]
    Fs = [compute_Fs(θs[1])]
    Gs = [compute_Gs(Fs[1])]
    αs = []

    t = 0
    converged = false

    while !converged

        # Generate new inflation factor
        φs = [0.5sum((L_e * (d_obs - Gs[end][:, i])).^2) for i ∈ 1:Ne]
        α_i = min(max(NG / 2mean(φs), √(NG / 2var(φs))), 1-t)^-1

        @info "Data-misfit mean: $(NG / 2mean(φs))"
        @info "Data-misfit variance: $(√(NG / 2var(φs)))"

        push!(αs, α_i) 
        t += α_i^-1
        if abs(t - 1.0) < 1e-8
            converged = true
        end 

        Δη = compute_Δs(ηs[end], Ne)
        ΔG = compute_Δs(Gs[end], Ne)

        C_GG = ΔG * ΔG'
        C_ηG = Δη * ΔG'

        # TODO: covariance localisation

        es = rand(MvNormal(αs[end] * Γ_e), Ne) # TODO: tidy up

        ηs_i = ηs[end] + C_ηG * inv(C_GG + αs[end] * Γ_e) * (d_obs .+ es - Gs[end] .- μ_e)
        θs_i = compute_θs(ηs_i)
        Fs_i = compute_Fs(θs_i)
        Gs_i = compute_Gs(Fs_i)

        push!(ηs, ηs_i)
        push!(θs, θs_i)
        push!(Fs, Fs_i)
        push!(Gs, Gs_i)

        if verbose
            @info "Iteration $(length(αs)) complete. t = $(t)."
        end

    end

    return ηs, θs, Fs, Gs, αs

end

# TODO: use these in EKI and EnRML
compute_θs(ηs, pr) = hcat([transform(pr, η_i) for η_i ∈ eachcol(ηs)]...)
compute_Fs(θs, F) = hcat([F(θ_i) for θ_i ∈ eachcol(θs)]...)
compute_Gs(Fs, G) = hcat([G(F_i) for F_i ∈ eachcol(Fs)]...)

function run_eks(
    F::Function,
    G::Function,
    pr::MaternField,
    d_obs::AbstractVector,
    μ_e::AbstractVector,
    Γ_e::AbstractMatrix,
    L_e::AbstractMatrix,
    Ne::Int,
    verbose::Bool=true
)

    NG = length(d_obs)

    ηs = [rand(pr, Ne)]
    θs = [compute_θs(ηs[1], pr)]
    Fs = [compute_Fs(θs[1], F)]
    Gs = [compute_Gs(Fs[1], G)]
    Δts = []

    t = 0
    converged = false 

    while !converged

        C_ηη = cov(ηs[end], dims=2)

        # TODO: compute timestep
        
        A_split = I + Δt * C_ηη
        B_split = ηs[end] - Δt * (1.0 / Ne) * ... * Γ_e_inv * 

        √(2 * Δts[end])


    end


end