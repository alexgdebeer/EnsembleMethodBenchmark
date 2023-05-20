"""Re-scales a set of log-weights such that the largest log-weight is 0.""" 
function rescale_logws(logws)
    return logws .- maximum(logws)
end

"""Normalises a set of weights such that they add to 1."""
function normalise_ws(ws)
    return ws ./ sum(ws)
end

"""Carries out a resampling procedure using a set of weights. Returns the 
indices that are samples."""
function resample_ws(ws)
    n = length(ws)
    cum_ws = cumsum(ws)
    r = rand()/n
    return [findfirst(cum_ws .≥ r+(i-1)/n) for i ∈ 1:n]
end

"""Calculates the effective sample size (ESS) associated with a set of 
weights."""
function ess(ws)
    return 1 ./ sum(ws.^2)
end

"""Returns the truncated singular value decomposition of a matrix A, 
under the requirement that the total energy retained is no less than a given 
amount."""
function tsvd(A::AbstractMatrix; energy=0.999)

    U, Λ, V = LinearAlgebra.svd(A)
    total_energy = sum(Λ.^2)

    for i ∈ 1:length(Λ)
        if sum(Λ[1:i].^2) / total_energy ≥ energy 
            return U[:, 1:i], Λ[1:i], V[:, 1:i]
        end
    end

    error("There is an issue in the TSVD function.")

end

"""Returns the inverse of a matrix, rescaled and then inverted using a 
truncated singular value decomposition."""
function inv_tsvd(A::AbstractMatrix; energy=0.999)

    size(A, 1) != size(A, 2) && error("Matrix is not square.")

    # Scale the matrix
    vars = LinearAlgebra.diag(A)
    stds_i = LinearAlgebra.Diagonal(1 ./ sqrt.(vars))
    A = stds_i * A * stds_i

    # Compute the TSVD of the scaled matrix 
    U, Λ, V = tsvd(A, energy=energy)

    # Form the inverse of the matrix
    A_i = stds_i * V * LinearAlgebra.Diagonal(1.0 ./ Λ) * U' * stds_i 

    return A_i

end

function kalman_gain(
    θs::AbstractMatrix, 
    ys::AbstractMatrix,
    Γ_ϵ::AbstractMatrix
)::AbstractMatrix

    Δθ = θs .- mean(θs, dims=2)
    Δy = ys .- mean(ys, dims=2)
    
    Γ_θy = Δθ*Δy'/(N_e-1)
    Γ_y  = Δy*Δy'/(N_e-1)

    return Γ_θy * inv_tsvd(Γ_y + Γ_ϵ)

end