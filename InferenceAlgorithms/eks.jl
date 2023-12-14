function run_eks(
    F::Function,
    G::Function,
    pr::MaternField,
    d_obs::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    Ne::Int;
    Δt₀::Real=2.0,
    t_stop::Real=2.0
)

    println("It. | Δt       | t        | misfit ")

    NG = length(d_obs)

    ηs = []
    θs = []
    Fs = []
    Gs = []

    ηs_i = rand(pr, Ne)
    θs_i, Fs_i, Gs_i = @time run_ensemble(ηs_i, F, G, pr)

    push!(ηs, ηs_i)
    push!(θs, θs_i)
    # push!(Fs, Fs_i)
    push!(Gs, Gs_i)

    t = 0
    i = 1
    while true

        μ_G = mean(Gs_i, dims=2)
        μ_η = mean(ηs_i, dims=2)
       
        C_ηη = cov(ηs[end], dims=2) # TODO: localisation / sampling error correction?
        D = (1.0 / Ne) * (Gs_i .- μ_G)' * (C_e \ (Gs_i .+ μ_e .- d_obs))
        ζ = rand(MvNormal(C_ηη + 1e-8 * I), Ne)
        
        Δt = Δt₀ / (norm(D) + 1e-6)
        t += Δt

        μ_misfit = mean(abs.(Gs[end] .+ μ_e .- d_obs))
        @printf "%3i | %.2e | %.2e | %.2e \n" i Δt t μ_misfit
        
        A_n = I + Δt * C_ηη
        B_n = ηs_i - 
            Δt * (ηs_i .- μ_η) * D +
            Δt * ((NG + 1) / Ne) * (ηs_i .- μ_η)

        ηs_i = @time (A_n \ B_n) + √(2 * Δt) * ζ
        θs_i, Fs_i, Gs_i = @time run_ensemble(ηs_i, F, G, pr)

        push!(ηs, ηs_i)
        push!(θs, θs_i)
        # push!(Fs, Fs_i)
        push!(Gs, Gs_i)

        i += 1
        if t ≥ t_stop 
            return ηs, θs, Fs, Gs
        end

    end

end