"""
    Taiga.FastDiagonalization(model::PotentialFlow{Dim, T}; method::Type{<:DataApproximationMethod} = Wachspress, kwargs)

Potential flow fast diagonalization preconditioner.
"""
function Taiga.FastDiagonalization(model::PotentialFlow{Dim,T}; method::Type{<:DataApproximationMethod}=Wachspress, kwargs...) where {Dim,T}
    FastDiagonalization(method, model; kwargs...)
end

function Taiga.FastDiagonalization(::Type{<:Wachspress}, model::PotentialFlow{Dim,T}; niter::Int=2) where {Dim,T}
    x = CartesianProduct(ntuple(k -> get_system_matrix_quadrule(model.S[k], model.S[k]).x, Dim)...)
    @evaluate C = PullbackBody(model.F)(x)
    data = approximate_patch_data(C; method=Wachspress, niter=niter)

    ∇u = ∇v = (k -> ι(k; dim=Dim))
    ∫ = KroneckerFactory(model.S, model.S)
    for α = 1:Dim
        ∫(∇u(α), ∇v(α); data=data[α, α])
    end
    L = ∫.data.K

    M = Vector{Matrix{T}}(undef, Dim)
    K = Vector{Matrix{T}}(undef, Dim)
    for α = 1:Dim
        a = (Dim + 1) - α
        b = (((a + 1) > Dim) ? 1 : (a + 1))
        β = (((α - 1) < 1) ? Dim : (α - 1))

        K[α] = L[α].data[a]
        M[β] = L[α].data[b]
    end

    λ = Vector{Vector{T}}(undef, Dim)
    u = Vector{Matrix{T}}(undef, Dim)
    for α = 1:Dim 
        decomposition = eigen(Symmetric(K[α]), Symmetric(M[α]))
        λ[(Dim+1)-α] = decomposition.values
        u[(Dim+1)-α] = decomposition.vectors
    end

    Λ = Diagonal(KroneckerSum(λ...))
    Λ⁻¹ = Diagonal(inv.(Λ.diag))
    U = KroneckerProduct(u...)
    return Taiga.FastDiagonalization(Λ⁻¹, U)
end