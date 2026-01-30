"""
    PotentialFlow{Dim, T, M <: GeometricMapping{Dim}, V <: ScalarSplineSpace{Dim, T}, W <: Field{Dim}}

Galerkin formulation of [potential flow](https://en.wikipedia.org/wiki/Potential_flow).
The solution field can be either velocity potential or stream function.

# Fields:
- `F::M`: geometric mapping
- `S::V`: constrained solution space
- `Δ::Partition{Dim, T}`: partition
- `uʰ::W`: solution field
- `ūʰ::W`: function satisfying Dirichlet boundary conditions
- `t::Function`: traction vector
"""
struct PotentialFlow{Dim,T,M<:GeometricMapping{Dim},V<:ScalarSplineSpace{Dim,T},W<:Field{Dim}} <: Model{Dim,T}
    F::M
    S::V
    Δ::Partition{Dim,T}
    uʰ::W
    ūʰ::W
    t::Function
    function PotentialFlow(F::M, S::V, C::ScalarSplineSpaceConstraints{Dim}, uʰ::W, ūʰ::W, t::Function) where {Dim,T,M<:GeometricMapping{Dim},V<:ScalarSplineSpace{Dim,T},W<:Field{Dim}}
        @assert domain(F) == domain(uʰ[1])
        Δ = Partition(S)
        S = ScalarSplineSpace(S, C)
        args = ι(0, dim=Dim)
        @assert applicable(t, args...)
        new{Dim,T,M,V,W}(F, S, Δ, uʰ, ūʰ, t)
    end
end

"""
    LinearOperator{Dim, T, B <: KroneckerProduct{T}, M <: AbstractMatrix{T}, V <: AbstractVector{T}}

Linear operator for [`PotentialFlow`](@ref) model.

# Fields:
- `C::B`: Kronecker product extraction operator
- `K::M`: sparse stiffness matrix
- `b::V`: forcing vector
- `cache₁::Vector{T}`: `mul!` cache
- `cache₂::Vector{T}`: `mul!` cache
"""
struct LinearOperator{Dim,T,B<:KroneckerProduct{T},S<:SparseMatrixCSC{T},V<:AbstractVector{T}} <: Taiga.LinearOperator{Dim,T}
    C::B
    K::S
    b::V
    cache₁::Vector{T}
    cache₂::Vector{T}
    function LinearOperator(model::P) where {Dim,T,P<:PotentialFlow{Dim,T}}
        F, t, Δ, space = model.F, model.t, model.Δ, model.S
        C = extraction_operator(space)

        quadrule = TensorProduct((d, u) -> PatchRule(d; npoints=ceil(Int, Degree(u) + 1), method=Legendre), Δ, space)
        acc = ElementAccessor(testspace=space, trialspace=space, quadrule=quadrule, incorporate_weights_in_testfuns=true)

        pullback_body = PullbackBody(F)
        pullback_boundary = ntuple(side -> PullbackBoundary(F, t, side), 2Dim)

        K = assemble_body(acc, Δ, pullback_body)
        b = assemble_boundary(acc, Δ, pullback_boundary)

        cache₁ = zeros(size(K,1))
        cache₂ = similar(cache₁)

        S, V, W, B = typeof(K), typeof(b), typeof(model.ūʰ), typeof(C)
        new{Dim,T,B,S,V}(C, K, b, cache₁, cache₂)
    end
end

Base.size(L::LinearOperator) = ntuple(k -> size(L.C, 2), 2)

"""
    Taiga.linear_operator(model::PotentialFlow)

Construct linear operator.
"""
Taiga.linear_operator(model::PotentialFlow) = LinearOperator(model)

function Taiga.LinearMaps._unsafe_mul!(b::AbstractVecOrMat, L::LinearOperator, v::AbstractVector)
    mul!(L.cache₁, L.C, v)
    mul!(L.cache₂, L.K, L.cache₁)
    mul!(b, L.C', L.cache₂)
    return b
end

"""
    Taiga.forcing!(b::AbstractVector, L::LinearOperator, model::PotentialFlow)

Update forcing vector.

# Arguments:
- `b`: cache vector
- `L`: linear operator
- `model`: model
"""
function Taiga.forcing!(b::AbstractVector, L::LinearOperator, model::PotentialFlow)
    b .= L.C' * L.b
    b .-= L.C' * (L.K * view(model.ūʰ[1].coeffs, :)) 
end

"""
    Taiga.forcing(L::LinearOperator, model::PotentialFlow)

Return forcing vector.
"""
function Taiga.forcing(L::LinearOperator, model::PotentialFlow)
    b = zeros(size(L, 1))
    forcing!(b, L, model)
end

"""
    Taiga.apply_particular_solution(L::LinearOperator, model::PotentialFlow, x₀::V)

Return particular solution.

# Arguments:
- `L`: linear operator
- `model`: model
- `x₀`: homogeneous solution
"""
function Taiga.apply_particular_solution(L::LinearOperator, model::PotentialFlow, x₀::V) where {V<:AbstractVector}
    model.ūʰ[1].coeffs[:] + view(L.C * x₀, :)
end
