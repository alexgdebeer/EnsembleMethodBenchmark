function run_eks(
    F::Function,
    G::Function,
    pr::MaternField,
    y::AbstractVector,
    μ_e::AbstractVector,
    C_e::AbstractMatrix,
    Ne::Int;
    Δt₀::Real=2.0,
    t_stop::Real=2.0
)

    println("It. | Δt       | t        | misfit ")

    NG = length(y)

    θs = []
    us = []
    Fs = []
    Gs = []

    θs_i = rand(pr, Ne)
    us_i, Fs_i, Gs_i = @time run_ensemble(θs_i, F, G, pr)

    push!(θs, θs_i)
    push!(us, us_i)
    # push!(Fs, Fs_i)
    push!(Gs, Gs_i)

    t = 0
    i = 1
    while true

        μ_G = mean(Gs_i, dims=2)
        μ_θ = mean(θs_i, dims=2)
       
        C_θθ = cov(θs[end], dims=2) # TODO: localisation / sampling error correction?
        D = (1.0 / Ne) * (Gs_i .- μ_G)' * (C_e \ (Gs_i .+ μ_e .- y))
        ζ = rand(MvNormal(C_θθ + 1e-8 * I), Ne)
        
        Δt = Δt₀ / (norm(D) + 1e-6)
        t += Δt

        μ_misfit = mean(abs.(Gs[end] .+ μ_e .- y))
        @printf "%3i | %.2e | %.2e | %.2e \n" i Δt t μ_misfit
        
        A_n = I + Δt * C_θθ
        B_n = θs_i - 
            Δt * (θs_i .- μ_θ) * D +
            Δt * ((NG + 1) / Ne) * (θs_i .- μ_θ)

        θs_i = @time (A_n \ B_n) + √(2 * Δt) * ζ
        us_i, Fs_i, Gs_i = @time run_ensemble(θs_i, F, G, pr)

        push!(θs, θs_i)
        push!(us, us_i)
        # push!(Fs, Fs_i)
        push!(Gs, Gs_i)

        i += 1
        if t ≥ t_stop 
            return θs, us, Fs, Gs
        end

    end

end