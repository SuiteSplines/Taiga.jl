"""
    Taiga.FastDiagonalization(model::Poisson{Dim,T}; method::Type{<:DataApproximationMethod}=Wachspress, kwargs...) where {Dim,T}

Constructs a [`FastDiagonalization`](@ref) preconditioner using a particular data approximation
method. [`Poisson`](@ref) supports [`Wachspress`](@ref) and [`ModalSplines`](@ref) data
approximation methods.
"""
function Taiga.FastDiagonalization(model::Poisson{Dim,T}; method::Type{<:DataApproximationMethod}=Wachspress, kwargs...) where {Dim,T}
    FastDiagonalization(method, model; kwargs...)
end


"""
    Taiga.FastDiagonalization(::Type{<:ModalSplines}, model::Poisson; spaces::NTuple{Dim, SplineSpace{T}})

Constructs a fast diagonalization preconditioner using rank 1 modal splines approximation
of model patch data on univariate spline spaces in `spaces`.
"""
function Taiga.FastDiagonalization(::Type{<:ModalSplines}, model::Poisson{Dim,T}; spaces::NTuple{Dim,SplineSpace{T}}) where {Dim,T}
    # check domain of model space and spaces in each direction
    @assert all(domain(model.F).data .== domain.(spaces))

    # evaluate patch data at grevillepoints of approximation spaces
    X = CartesianProduct(grevillepoints.(spaces)...)
    C = pullback_body_data(model.F, X, model.κ)

    # compute modal weighting splines of rank 1
    data = approximate_patch_data(C; method=ModalSplines, spaces=spaces)

    # assemble Kronecker approximation
    ∇u = ∇v = (k -> ι(k; dim=Dim))
    ∫ = KroneckerFactory(model.S, model.S)
    for α = 1:Dim
        ∫(∇u(α), ∇v(α); data=data[1, α, α])
    end
    L = ∫.data.K

    # collect univariate mass and stiffness matrices weighted by modal splines
    M = Vector{Matrix{T}}(undef, Dim)
    K = Vector{Matrix{T}}(undef, Dim)
    for α = 1:Dim # parametric direction α for univariate stiffness in L[α]
        a = (Dim + 1) - α # data index of univariate stiffness in L[α] for derivative index α
        b = (((a + 1) > Dim) ? 1 : (a + 1)) # data index of a univariate mass in L[α] (using cyclic ordering)
        β = (((α - 1) < 1) ? Dim : (α - 1)) # parametric direction β for the univariate mass in L[α] at data index b

        # Eigenvectors are not sign definite: while this is not a problem within
        # the approximation itself (the sign cancels out with the KroneckerProduct)
        # FastDiagonalization relies on an generalized eigenvalue decomposition.
        # Thus if !isposdef(M[β]) then flip sign of current mass and stiffness matrix
        sign = !isposdef(Symmetric(L[α].data[b])) ? -1 : 1
        K[α] = sign * L[α].data[a]
        M[β] = sign * L[α].data[b]
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
    return Taiga.FastDiagonalization(Λ⁻¹, U)
end

"""
    Taiga.FastDiagonalization(::Type{<:Wachspress}, model::Poisson; niter::Int)

Constructs a fast diagonalization preconditioner using `niter` of Wachspress
algorithm to approximate patch data.
"""
function Taiga.FastDiagonalization(::Type{<:Wachspress}, model::Poisson{Dim,T}; niter::Int=2) where {Dim,T}
    # compute Wachspress approximation
    X = CartesianProduct(ntuple(k -> get_system_matrix_quadrule(model.S[k], model.S[k]).x, Dim)...)
    C = PoissonModule.pullback_body_data(model.F, X, model.κ)
    data = approximate_patch_data(C; method=Wachspress, niter=niter)

    # assemble Kronecker approximation
    ∇u = ∇v = (k -> ι(k; dim=Dim))
    ∫ = KroneckerFactory(model.S, model.S)
    for α = 1:Dim
        ∫(∇u(α), ∇v(α); data=data[α, α])
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
    return Taiga.FastDiagonalization(Λ⁻¹, U)
end