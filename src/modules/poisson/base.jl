"""
    Poisson{Dim,T,M<:GeometricMapping{Dim},V<:ScalarSplineSpace{Dim,T},W<:Field{Dim}} <: Model{Dim,T}

## Strong form

    -∇ ⋅ (κ∇u) = f      in Ω
             u = ū      on Γ₁
        ∇u ⋅ n = t ⋅ n  on Γ₂

## Weak form

    Find the homogeneous solution uₒ ∈ Sʰ, s.t. ∀v ∈ Sʰ

        ∫ ∇(uₒ - ū) ⋅ ∇v dΩ - ∫ (t ⋅ n) v dΓ₁ = - ∫ f v dΓ

    for the homogeneous problem uₒ = 0 on Γ. The solution
    u ∈ ū ⊕ Sʰ satisfying nonhomogeneous Dirichlet boundary
    conditions is u = uₒ + ū.

## Note

The linear operators [`Taiga.PoissonModule.LinearOperator`](@ref) and
[`Taiga.PoissonModule.MatrixfreeLinearOperator`](@ref) correspond to
the homogeneous problem on the constrained space.
The solution of the nonhomogeneous problem is obtained by
[`Taiga.PoissonModule.apply_particular_solution`](@ref).

# Fields:
- `F::M`: geometric mapping
- `S::V`: solution space
- `Δ::Partition{Dim,T}`: partition of the space
- `uʰ::Field`: solution field
- `ūʰ::W`: scalar solution field satisfying Dirichlet boundary conditions
- `κ::Function`: conductivity matrix of size `Dim × Dim` as function of spatial coordinates
- `f::Function`: body force as function of spatial coordinates
- `t::Function`: traction as function of spatial coordinates
"""
struct Poisson{Dim,T,M<:GeometricMapping{Dim},V<:ScalarSplineSpace{Dim,T},W<:Field{Dim}} <: Model{Dim,T}
    F::M # geometric mapping
    S::V # solution space
    Δ::Partition{Dim,T} # partition
    uʰ::W # solution field
    ūʰ::W # field satisfying boundary conditions
    κ::Function # conductivity
    f::Function # bodyforce
    t::Function # traction

    @doc """
        Poisson(F::M, S::V, C::ScalarSplineSpaceConstraints{Dim}, uʰ::W, ūʰ::W, κ::Function, f::Function, t::Function; matrixfree=false) where {Dim,T,M<:GeometricMapping{Dim},V<:ScalarFunctionSpace{Dim,T},W<:Field{Dim}}

    Construct [`Poisson`](@ref) model.

    # Arguments:
    - `F::M`: geometric mapping
    - `S::V`: scalar function space
    - `C::ScalarSplineSpaceConstraints{Dim}`: function space constraints
    - `uʰ::W`: scalar solution field
    - `ūʰ::W`: scalar solution field satisfying Dirichlet boundary conditions
    - `κ::Function`: conductivity matrix as a function of spatial coordinates 
    - `f::Function`: body force as function of spatial coordinates
    - `t::Function`: traction vector as function of spatial coordinates
    - `matrixfree::Bool`: boolean flag for matrix-free forward problem, default is `false`
    """
    function Poisson(F::M, S::V, C::ScalarSplineSpaceConstraints{Dim}, uʰ::W, ūʰ::W, κ::Function, f::Function, t::Function) where {Dim,T,M<:GeometricMapping{Dim},V<:ScalarSplineSpace{Dim,T},W<:Field{Dim}}
        # check if mapping and solution field share domain
        @assert domain(F) == domain(uʰ[1])

        # check nargs = Dim for functions κ, f and t
        args = ι(0, dim=Dim)
        @assert applicable(κ, args...)
        @assert applicable(f, args...)
        @assert applicable(t, args...)

        Δ = Partition(S)

        # set S to function space satisfying homogeneous boundary conditions
        S = ScalarSplineSpace(S, C)

        new{Dim,T,M,V,W}(F, S, Δ, uʰ, ūʰ, κ, f, t)
    end
end

"""
    LinearOperator{Dim,T,M<:AbstractMatrix{T},V<:AbstractVector{T},B<:KroneckerProduct{T}} <: Taiga.LinearOperator{Dim,T}

Linear operator for the homogeneous system in [`Poisson`](@ref) model.

# Fields:
- `K::M`: stiffness matrix
- `b::V`: rhs vector
- `C::B`: Kronecker product extraction operator
"""
struct LinearOperator{Dim,T,M<:AbstractMatrix{T},V<:AbstractVector{T},W<:Field{Dim},B<:KroneckerProduct{T}} <: Taiga.LinearOperator{Dim,T}
    K::M # stiffness matrix
    b::V # rhs vector
    ūʰ::W # field satisfying boundary conditions
    C::B # Kronecker product extraction operator
    function LinearOperator(model::P) where {Dim,T,P<:Poisson{Dim,T}}
        # just for convenience
        space, F, κ, f, t, Δ = model.S, model.F, model.κ, model.f, model.t, model.Δ

        # quadrule
        quadrule = TensorProduct((d, u) -> PatchRule(d; npoints=ceil(Int, Degree(u) + 1), method=Legendre), Δ, space)

        # Accessor
        acc = ElementAccessor(testspace=space, trialspace=space, quadrule=quadrule, incorporate_weights_in_testfuns=true)

        # pullback body
        pullback_body!(e) = pullback_body_data!(acc, e, FieldEvaluationCache(acc, Dim, Dim), FieldEvaluationCache(acc, 1, 1), F, κ, f)

        # pullback boundary
        pullback_boundary!(e, side) = pullback_boundary_data!(acc, e, FieldEvaluationCache(acc), boundary(F, side), t)

        # assemble body
        K, b, C = assemble_body(; space=model.S, partition=Δ, acc=acc, pullback_body=pullback_body!)

        # update rhs by boundary contributions
        assemble_boundary!(b; partition=Δ, acc=acc, mapping=F, pullback_boundary=pullback_boundary!, traction=t)

        # construct linear operator
        M, V, W, B = typeof(K), typeof(b), typeof(model.ūʰ), typeof(C)
        new{Dim,T,M,V,W,B}(K, b, model.ūʰ, C)
    end
end

Base.size(L::LinearOperator) = ntuple(k -> size(L.C, 2), 2)

function Taiga.LinearMaps._unsafe_mul!(b::AbstractVecOrMat, L::LinearOperator, v::AbstractVector)
    a = L.C * v
    a = L.K * a
    mul!(b, L.C', a)
    return b
end

"""
    MatrixfreeLinearOperator{Dim, T, M <: Poisson{Dim, T}, A <: Accessor{Dim}, B <: KroneckerProduct{T}, V <: VectorSumfactoryCache{Dim}} <: Taiga.LinearOperator{Dim,T}

Matrix-free linear operator for the homogeneous system in [`Poisson`](@ref) model.

# Fields:
- `model::M`: model
- `acc::A`: patch accessor
- `C::B`: Kronecker product extraction operator
- `pullback_body::Function`: pullback for the body terms
- `sumfact_cache_u::V`: sum factorization vector cache
"""
struct MatrixfreeLinearOperator{Dim,T,M<:Poisson{Dim,T},A<:Accessor{Dim},B<:KroneckerProduct{T},V<:VectorSumfactoryCache{Dim}} <: Taiga.LinearOperator{Dim,T}
    model::M
    acc::A
    C::B
    pullback_body::Function
    sumfact_cache_u::V
    function MatrixfreeLinearOperator(model::P; constrained::Bool=true) where {Dim,T,P<:Poisson{Dim,T}}
        # just for convenience
        F, κ, f, t, Δ = model.F, model.κ, model.f, model.t, model.Δ

        # get space
        space = constrained ? model.S : ScalarSplineSpace(model.S)

        # space constraints
        C = extraction_operator(model.S)

        # quadrule
        quadrule = TensorProduct((d, u) -> PatchRule(d; npoints=ceil(Int, Degree(u) + 1), method=Legendre), Δ, space)

        # Accessor
        acc = PatchAccessor(testspace=space, trialspace=space, quadrule=quadrule, incorporate_weights_in_testfuns=true)

        # create caches for sum factorization
        sumfact_cache_u = VectorSumfactoryCache(acc)

        # precompute mapping data
        X = QuadraturePoints(acc)
        @evaluate Y = F(X)
        @evaluate ∇Y = Gradient(F)(X)

        # an @evaluate's lazy evaluation quirk: since Y and ∇Y are supposed to be persistent
        # throughout the life of MatrixfreeLinearOperator, they need a deep copy! Otherwise
        # a next call @evaluate Y₂ = F(X) for different X *will* cause havoc and mayhem
        Y, ∇Y = copy(Y), copy(∇Y)

        # pullback body
        pullback_body!(u) = pullback_body_data_matrixfree!(acc, FieldEvaluationCache(acc, Dim, 1), Y, ∇Y, u, κ)

        # construct linear operator
        A, B, V = typeof(acc), typeof(C), typeof(sumfact_cache_u)
        new{Dim,T,P,A,B,V}(model, acc, C, pullback_body!, sumfact_cache_u)
    end
end

function isconstrained(L::MatrixfreeLinearOperator{Dim}) where {Dim}
    trialfuns = L.acc.trialfuns.data
    m, n = 1, 1
    for k in Base.OneTo(Dim)
        m *= size(trialfuns[k].C, 1)
        n *= size(trialfuns[k].C, 2)
    end
    m != n
end
Base.size(L::MatrixfreeLinearOperator) = isconstrained(L) ? ntuple(k -> size(L.C, 2), 2) : ntuple(k -> size(L.C, 1), 2)
@inline apply_coeffs(L::MatrixfreeLinearOperator{Dim,T}, v::AbstractVector{T}) where {Dim,T} = apply_coeffs(L, v, Val(isconstrained(L)))
@inline apply_coeffs(L::MatrixfreeLinearOperator{Dim,T}, v::AbstractVector{T}, ::Val{true}) where {Dim,T} = mul!(view(L.model.uʰ[1].coeffs, :), L.C, v)
@inline apply_coeffs(L::MatrixfreeLinearOperator{Dim,T}, v::AbstractVector{T}, ::Val{false}) where {Dim,T} = view(L.model.uʰ[1].coeffs, :) .= v

function Taiga.LinearMaps._unsafe_mul!(b::AbstractVecOrMat, L::M, v::AbstractVector) where {Dim,T,M<:MatrixfreeLinearOperator{Dim,T}}
    # get solution field
    u = L.model.uʰ

    # apply v to solution field
    apply_coeffs(L, v)

    # create integration object
    ∫ = Sumfactory(L.sumfact_cache_u)

    # pullback solution data at quadrature points
    D = L.pullback_body(u)

    # get test functions
    ∇v = ntuple(k -> TestFunctions(L.acc; ders=ι(k, dim=Dim)), Dim)

    # perform quadrature by sum factorization
    for k in Base.OneTo(Dim)
        ∫(∇v[k]; data=D.data[k], reset=false)
    end

    copyto!(view(b, :), ∫.data[1])
end

"""
    LinearOperatorApproximation{Dim,T} <: Taiga.LinearOperatorApproximation{Dim,T}

Linear operator approximation for the homogeneous system in [`Poisson`](@ref) model.

# Fields:
- `L::KroneckerProductAggregate{T}`: Kronecker product aggregate
"""
struct LinearOperatorApproximation{Dim,T} <: Taiga.LinearOperatorApproximation{Dim,T}
    L::KroneckerProductAggregate{T} # KroneckerProductAggregate
end

Base.size(L::LinearOperatorApproximation) = size(L.L)
function Taiga.LinearMaps._unsafe_mul!(b::AbstractVecOrMat, L::LinearOperatorApproximation, v::AbstractVector)
    mul!(b, L.L, v)
end

function LinearOperatorApproximation(model::P, method::Type{<:ModalSplines}; spaces::NTuple{Dim,SplineSpace{T}}, rank::Int) where {Dim,T,P<:Poisson{Dim,T}}
    # compute modal weighting splines
    X = CartesianProduct(grevillepoints.(spaces)...)
    C = PoissonModule.pullback_body_data(model.F, X, model.κ)
    data = approximate_patch_data(C; method=method, spaces=spaces, rank=rank)

    ∇u = k -> ι(k, dim=Dim)
    ∇v = k -> ι(k, dim=Dim)

    # assemble Kronecker approximation
    ∫ = KroneckerFactory(model.S, model.S)
    for r in Base.OneTo(size(data, 1))
        for β in Base.OneTo(Dim)
            for α in Base.OneTo(Dim)
                ∫(∇u(α), ∇v(β); data=data[r, α, β])
            end
        end
    end
    LinearOperatorApproximation{Dim,T}(∫.data)
end

function LinearOperatorApproximation(model::P, method::Type{<:CanonicalPolyadic}; tol::T=10e-1, rank::Int, ntries::Int=10) where {Dim,T,P<:Poisson{Dim,T}}
    # compute modal weighting splines
    X = CartesianProduct(ntuple(k -> get_system_matrix_quadrule(model.S[k], model.S[k]).x, Dim)...)
    C = PoissonModule.pullback_body_data(model.F, X, model.κ)
    data = approximate_patch_data(C; method=method, rank=rank, tol=tol, ntries=ntries)

    ∇u = k -> ι(k, dim=Dim)
    ∇v = k -> ι(k, dim=Dim)

    # assemble Kronecker approximation
    ∫ = KroneckerFactory(model.S, model.S)
    for r in Base.OneTo(size(data, 1))
        for β in Base.OneTo(Dim)
            for α in Base.OneTo(Dim)
                ∫(∇u(α), ∇v(β); data=data[r, α, β])
            end
        end
    end
    LinearOperatorApproximation{Dim,T}(∫.data)
end

function LinearOperatorApproximation(model::P, method::Type{<:Wachspress}; niter::Int) where {Dim,T,P<:Poisson{Dim,T}}
    # compute Wachspress approximation
    X = CartesianProduct(ntuple(k -> get_system_matrix_quadrule(model.S[k], model.S[k]).x, Dim)...)
    C = PoissonModule.pullback_body_data(model.F, X, model.κ)
    data = approximate_patch_data(C; method=method, niter=niter)

    ∇u = k -> ι(k, dim=Dim)
    ∇v = k -> ι(k, dim=Dim)

    # assemble Kronecker approximation
    ∫ = KroneckerFactory(model.S, model.S)
    for α in Base.OneTo(size(data, 1))
        ∫(∇u(α), ∇v(α); data=data[α, α])
    end
    LinearOperatorApproximation{Dim,T}(∫.data)
end

"""
    linear_operator(model::Poisson; matrixfree::Bool=false, constrained::Bool=true)

Returns a linear operator corresponding for `model`.
Depending on the `matrixfree` keyword either [`PoissonModule.LinearOperator`](@ref)
or [`PoissonModule.MatrixfreeLinearOperator`](@ref) is returned. For
[`PoissonModule.MatrixfreeLinearOperator`](@ref) the `constrained` flag controls
if the extraction operator is built into the matrix-free linear operator.
"""
Taiga.linear_operator(model::Poisson; matrixfree::Bool=false, constrained::Bool=true) = matrixfree ? MatrixfreeLinearOperator(model; constrained=constrained) : LinearOperator(model)

"""
    linear_operator_approximation(model::Poisson; method::Type{<:DataApproximationMethod})

Returns [`PoissonModule.LinearOperatorApproximation`](@ref) for [`Poisson`](@ref) model
using one of [`Taiga.DataApproximationMethod`](@ref).
"""
function Taiga.linear_operator_approximation(model::Poisson; method::Type{<:DataApproximationMethod}, kwargs...)
    LinearOperatorApproximation(model, method; kwargs...)
end

"""
    forcing(L::LinearOperator)

Returns forcing vector.
"""
function Taiga.forcing(L::LinearOperator)
    b = zeros(size(L, 1))
    forcing!(b, L)
end

"""
    forcing!(b::AbstractVector, L::LinearOperator)

Updates forcing vector `b`.
"""
function Taiga.forcing!(b::AbstractVector, L::LinearOperator)
    b .= L.C' * L.b # forcing terms
    b .-= L.C' * (L.K * view(L.ūʰ[1].coeffs, :)) # nonhomogeneous Dirichlet bc
end

function Taiga.forcing!(b::Vector{T}, model::M) where {Dim,T,M<:Poisson{Dim,T}}
    # just for convenience
    space, Cᵗ, F, κ, f, t, Δ = model.S, extraction_operator(model.S)', model.F, model.κ, model.f, model.t, model.Δ

    # linear operator without constraints
    A = MatrixfreeLinearOperator(model; constrained=false)

    # cache for boundary terms
    b_Ω = zeros(size(Cᵗ, 1))
    b_nh = zeros(size(Cᵗ, 2))
    b_∂Ω = zeros(size(Cᵗ, 2))

    # quadrule
    quadrule = TensorProduct((d, u) -> PatchRule(d; npoints=ceil(Int, Degree(u) + 1), method=Legendre), Δ, space)

    # Accessor
    acc_element = ElementAccessor(testspace=space, trialspace=space, quadrule=quadrule, incorporate_weights_in_testfuns=true)
    acc_patch = PatchAccessor(testspace=space, trialspace=space, quadrule=quadrule, incorporate_weights_in_testfuns=true)

    # pullback forcing
    pullback_forcing!() = pullback_forcing_data!(acc_patch, FieldEvaluationCache(acc_patch, 1, 1), F, κ, f)

    # pullback boundary
    pullback_boundary!(e, side) = pullback_boundary_data!(acc, e, FieldEvaluationCache(acc), boundary(F, side), t)

    # assemble forcing
    assemble_forcing!(b_Ω; space=model.S, partition=Δ, acc=acc_patch, pullback_forcing=pullback_forcing!)

    # update rhs by boundary contributions
    assemble_boundary!(b_∂Ω; partition=Δ, acc=acc_element, mapping=F, pullback_boundary=pullback_boundary!, traction=t)

    # apply constraints
    mul!(b, Cᵗ, b_∂Ω)
    b .+= b_Ω

    # subtract non-homogeneous boundary contribution
    mul!(b_nh, A, view(model.ūʰ[1].coeffs, :))
    mul!(b_Ω, Cᵗ, b_nh)
    b .-= b_Ω
end

function Taiga.forcing(model::M) where {Dim,T,M<:Poisson{Dim,T}}
    b = zeros(dimension(model.S))
    forcing!(b, model)
    return b
end

"""
    apply_particular_solution(L::LinearOperator, x₀::V) where {V<:AbstractVector}

Applies particular solution to homogeneous solution vector.
"""
function Taiga.apply_particular_solution(L::LinearOperator, x₀::V) where {V<:AbstractVector}
    L.ūʰ[1].coeffs[:] + view(L.C * x₀, :)
end

"""
    apply_particular_solution(model::Poisson, x₀::V) where {V<:AbstractVector}

Applies particular solution to homogeneous solution vector.
"""
function Taiga.apply_particular_solution(model::Poisson, x₀::V) where {V<:AbstractVector}
    C = extraction_operator(model.S)
    model.ūʰ[1].coeffs[:] + view(C * x₀, :)
end


