function run_chain(
    f::Function,
    g::Function,
    π::Distribution,
    L::Distribution,
    K::Distribution,
    N::Int;
    verbose::Bool=true,
    θ_s::Union{AbstractVector,Nothing}=nothing
)

    N_θ = length(π.μ)
    N_y = length(L.μ)

    θs = zeros(N_θ, N)
    ys = zeros(N_y, N)

    # Sample a starting point from the prior
    θs[:,1] = θ_s === nothing ? rand(π) : θ_s
    ys[:,1] = g(f(θs[:,1]))

    j = 0

    for i ∈ 2:N 

        # Generate a proposal  
        θ_p = θs[:,i-1] + rand(K)

        # Run the forward model 
        y_p = g(f(θ_p))

        # Calculate the acceptance probability
        log_h = logpdf(π, θ_p) -
                logpdf(π, θs[:,i-1]) +
                logpdf(L, y_p) -
                logpdf(L, ys[:,i-1]) # +
                # logpdf(K, θs[:,i-1]-θ_p) -
                # logpdf(K, θ_p-θs[:,i-1])
        
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
            @info "$i iterations complete (α = $α%)."
        end
        
    end

    return θs, ys

end


function run_mcmc(
    f::Function,
    g::Function,
    π::Distribution,
    L::Distribution,
    K::Distribution,
    N::Int;
    n_chains::Int=1,
    verbose::Bool=true,
    θ_s::Union{AbstractVector,Nothing}=nothing
)

    @info "Starting MCMC..."

    N_θ = length(π.μ)
    N_y = length(L.μ)

    θs = zeros(N_θ, N, n_chains)
    ys = zeros(N_y, N, n_chains)

    Threads.@threads for i = 1:n_chains
        θs[:,:,i], ys[:,:,i] = run_chain(f, g, π, L, K, N, verbose=verbose, θ_s=θ_s)
    end

    return θs, ys

end