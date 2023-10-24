using LinearAlgebra

function run_ensemble(
    F::Function,
    G::Function,
    θs::AbstractArray,
    NF::Int,
    NG::Int,
    Ne::Int
)::Tuple{AbstractMatrix, AbstractMatrix}

    Fs = zeros(NF, Ne)
    Gs = zeros(NG, Ne)

    for i ∈ 1:Ne
        Fs[:,i] = F(θs[:,i])
        Gs[:,i] = G(Fs[:,i])
    end

    return Fs, Gs

end

function calculate_deviations(
    xs::AbstractMatrix, 
    N::Int
)::AbstractMatrix

    return (xs .- mean(xs, dims=2)) ./ √(N-1)

end

function run_eki_dmc(
    F::Function,
    G::Function,
    pr::MaternField, 
    y::AbstractVector,
    μ_e::AbstractVector,
    Γ_e::AbstractMatrix,
    NF::Int,
    Ne::Int,
    localisation::Bool=false, # TODO: add localisation options
    verbose::Bool=true
)

    NG = length(y)
    L_e = cholesky(inv(Γ_e)).U # TODO: move out of this function

    ηs = []
    Fs = []
    Gs = []
    αs = []

    ηs_i = rand(pr, Ne)
    Fs_i, Gs_i = run_ensemble(F, G, ηs_i, NF, NG, Ne)

    push!(ηs, ηs_i)
    push!(Fs, Fs_i)
    push!(Gs, Gs_i)

    t = 0
    converged = false

    while !converged

        # Generate new inflation factor
        φs = [0.5sum((L_e * (y - Gs[end][:, i])).^2) for i ∈ 1:Ne]
        α_i = min(max(NG / 2mean(φs), √(NG / 2var(φs))), 1-t)^-1
        push!(αs, α_i) 

        @info "Data-misfit mean: $(NG / 2mean(φs))"
        @info "Data-misfit variance: $(√(NG / 2var(φs)))"

        t += α_i^-1
        if abs(t - 1.0) < 1e-8
            converged = true
        end 

        Δη = calculate_deviations(ηs[end], Ne)
        ΔG = calculate_deviations(Gs[end], Ne)

        C_GG = ΔG * ΔG'
        C_ηG = Δη * ΔG'

        # TODO: covariance localisation

        es = rand(MvNormal(αs[end] * Γ_e), Ne) # TODO: tidy up

        ηs_i = ηs[end] + C_ηG * inv(C_GG + αs[end] * Γ_e) * (y .+ es - Gs[end] .- μ_e)
        Fs_i, Gs_i = run_ensemble(F, G, ηs_i, NF, NG, Ne)

        push!(ηs, ηs_i)
        push!(Fs, Fs_i)
        push!(Gs, Gs_i)

        if verbose
            @info "Iteration $(length(αs)) complete. t = $(t)."
        end

    end

    return ηs, Fs, Gs, αs

end