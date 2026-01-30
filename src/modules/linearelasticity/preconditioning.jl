"""
    Taiga.FastDiagonalization(model::Model{Dim, T}; method::Type{<:DataApproximationMethod} = Wachspress, kwargs)

Computes a block diagonal fast diagonalization preconditioner for [`Model`](@ref).

# Arguments:
- `model`: [`Model`](@ref) model definition
- `method`: method for data approximation
- `kwargs`: optional keyword arguments
"""
function Taiga.FastDiagonalization(model::Model{Dim,T}; method::Type{<:DataApproximationMethod}=Wachspress, kwargs...) where {Dim,T}
    FastDiagonalization(method, model; kwargs...)
end

function Taiga.FastDiagonalization(::Type{<:Wachspress}, model::Model{Dim,T}; niter::Int=2) where {Dim,T}
    # compute Wachspress approximation of diagonal blocks
    X = CartesianProduct(ntuple(k -> get_system_matrix_quadrule(model.S[1][k], model.S[1][k]).x, Dim)...)
    pullback = PullbackBilinearForm(model.F, model.material)
    @evaluate C = pullback(X)
    data = [approximate_patch_data(C[i,i]; method=Wachspress, niter=niter) for i in 1:Dim]

    # loop over blocks
    F = []
    for k in 1:Dim
        ∇u = ∇v = (k -> ι(k; dim=Dim))
        ∫ = KroneckerFactory(model.S[k], model.S[k])
        for α = 1:Dim
            ∫(∇u(α), ∇v(α); data=data[k][α, α])
        end
        L = ∫.data.K

        # collect univariate mass and stiffness matrices weighted by modal splines
        M = Vector{Matrix{T}}(undef, Dim)
        K = Vector{Matrix{T}}(undef, Dim)
        for α = 1:Dim # parametric direction α for univariate stiffness in L[α]
            a = (Dim + 1) - α # data index of univariate stiffness in L[α] for derivative index α
            b = (((a + 1) > Dim) ? 1 : (a + 1)) # data index of a univariate mass in L[α] (using cyclic ordering)
            β = (((α - 1) < 1) ? Dim : (α - 1)) # parametric direction β for the univariate mass in L[α] at data index b

            K[α] = L[α].data[a]
            M[β] = L[α].data[b]
        end

        # eigenvalues and eigenvector ordered by parametric direction α = 1, 2, ...
        λ = Vector{Vector{T}}(undef, Dim)
        u = Vector{Matrix{T}}(undef, Dim)
        for α = 1:Dim 
            decomposition = eigen(Symmetric(K[α]), Symmetric(M[α]))
    
            # K[α] and M[α] are ordered in reverse!
            λ[(Dim+1)-α] = decomposition.values
            u[(Dim+1)-α] = decomposition.vectors
        end

        Λ = Diagonal(KroneckerSum(λ...))
        Λ⁻¹ = Diagonal(inv.(Λ.diag))
        U = KroneckerProduct(u...)
        push!(F, Taiga.FastDiagonalization(Λ⁻¹, U))
    end
    BlockDiagonalPreconditioner{Dim,T}(LinearMaps.BlockDiagonalMap(F...))
end
