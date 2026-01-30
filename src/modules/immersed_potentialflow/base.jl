"""
    ImmersedPotentialFlow{Dim, T, M <: GeometricMapping{Dim}, V <: ScalarSplineSpace{Dim, T}, W <: Field{Dim}}

Immersed Galerkin formulation of [potential flow](https://en.wikipedia.org/wiki/Potential_flow).
The solution field can be either velocity potential or stream function.

# Fields:
- `F::M`: geometric mapping
- `S::V`: constrained solution space
- `Δ::Partition{Dim, T}`: partition
- `uʰ::W`: solution field
- `ūʰ::W`: function satisfying Dirichlet boundary condition
- `t::Function`: traction vector
- `ϕ::AlgoimCallLevelSetFunction`: level set function
"""
struct ImmersedPotentialFlow{Dim,T,M<:GeometricMapping{Dim},V<:ScalarSplineSpace{Dim,T},W<:Field{Dim}} <: Model{Dim,T}
    F::M
    S::V
    Δ::Partition{Dim,T}
    uʰ::W
    ūʰ::W
    t::Function
    ϕ::AlgoimCallLevelSetFunction
    function ImmersedPotentialFlow(F::M, S::V, C::ScalarSplineSpaceConstraints{Dim}, uʰ::W, ūʰ::W, t::Function, ϕ::AlgoimCallLevelSetFunction) where {Dim,T,M<:GeometricMapping{Dim},V<:ScalarSplineSpace{Dim,T},W<:Field{Dim}}
        @assert domain(F) == domain(uʰ[1])
        Δ = Partition(S)
        S = ScalarSplineSpace(S, C)

        args = ι(0, dim=Dim)
        @assert applicable(t, args...)
        @assert applicable(ϕ, zeros(Dim))

        new{Dim,T,M,V,W}(F, S, Δ, uʰ, ūʰ, t, ϕ)
    end
end

"""
    LinearOperator{Dim, T, S <: SparseMatrixCSC{T}, M <: AbstractMatrix{T}, V <: AbstractVector{T}}

Linear operator for [`ImmersedPotentialFlow`](@ref) model.

# Fields:
- `C::S`: sparse extraction operator
- `E::S`: sparse extension operator
- `K::M`: sparse stiffness matrix
- `L::S`: sparse linear operator `Eᵀ Cᵀ K C E`
- `b::V`: right hand side vector
"""
struct LinearOperator{Dim,T,S<:SparseMatrixCSC{T},V<:AbstractVector{T}} <: Taiga.LinearOperator{Dim,T}
    C::S
    E::S
    K::S
    L::S
    b::V
    function LinearOperator(model::P; show_progress::Bool=true) where {Dim,T,P<:ImmersedPotentialFlow{Dim,T}}
        F, t, Δ, space, ϕ = model.F, model.t, model.Δ, model.S, model.ϕ
        C = kron(reverse(ntuple(d -> space[d].C, Dim))...)
        E = spline_extension_operator(space, ϕ)

        quadrule = TensorProduct((d, u) -> PatchRule(d; npoints=ceil(Int, Degree(u) + 1), method=Legendre), Δ, space)
        acc = ElementAccessor(testspace=space, trialspace=space, quadrule=quadrule, incorporate_weights_in_testfuns=true)

        quadrule_immersed = CutcellQuadratureRule(partition=Δ, mapping=ϕ, npoints=max(map(s -> Degree(s), space)...)+1);
        acc_immersed = ElementAccessor(testspace=space, trialspace=space, quadrule=quadrule_immersed);

        pullback_body = PullbackBody(F)
        pullback_boundary = ntuple(side -> PullbackBoundary(F, t, side), 2Dim)

        K = assemble_body(acc, acc_immersed, Δ, pullback_body, ϕ; show_progress=show_progress)
        b = assemble_boundary(acc, acc_immersed, Δ, pullback_boundary, ϕ)

        L = E' * C' * K * C * E

        V, W, S = typeof(b), typeof(model.ūʰ), typeof(C)
        new{Dim,T,S,V}(C, E, K, L, b)
    end
end

Base.size(L::LinearOperator) = ntuple(k -> size(L.E, 2), 2)

"""
    Taiga.linear_operator(model::ImmersedPotentialFlow; show_progress::Bool = true)

Construct linear operator.
"""
Taiga.linear_operator(model::ImmersedPotentialFlow; show_progress::Bool=true) = LinearOperator(model; show_progress=show_progress)

function Taiga.LinearMaps._unsafe_mul!(b::AbstractVecOrMat, L::LinearOperator, v::AbstractVector)
    mul!(b, L.L, v)
    return b
end

"""
    SparseArrays.sparse(L::LinearOperator)

Get sparse linear operator.
"""
SparseArrays.sparse(L::LinearOperator) = L.L

"""
    Taiga.forcing!(b::AbstractVector, L::LinearOperator, model::ImmersedPotentialFlow)

Update forcing vector.

# Arguments:
- `b`: cache vector
- `L`: linear operator
- `model`: model
"""
function Taiga.forcing!(b::AbstractVector, L::LinearOperator, model::ImmersedPotentialFlow)
    b .= L.E' * L.C' * L.b
    b .-= L.E' * L.C' * (L.K * view(model.ūʰ[1].coeffs, :)) 
end

"""
    Taiga.forcing(L::LinearOperator, model::ImmersedPotentialFlow)

Return foring vector.
"""
function Taiga.forcing(L::LinearOperator, model::ImmersedPotentialFlow)
    b = zeros(size(L, 1))
    forcing!(b, L, model)
end

"""
    Taiga.apply_particular_solution(L::LinearOperator, model::ImmersedPotentialFlow, x₀::V)

Return particular solution.

# Arguments:
- `L`: linear operator
- `model`: model
- `x₀`: homogeneous solution
"""
function Taiga.apply_particular_solution(L::LinearOperator, model::ImmersedPotentialFlow, x₀::V) where {V<:AbstractVector}
    model.ūʰ[1].coeffs[:] + view(L.C * L.E * x₀, :)
end

"""
    embedding(width::T, height::T)

Return Cartesian background mapping.
"""
function embedding(width::T, height::T) where {T<:Real}
    @assert width > 0 && height > 0

    S₁ = SplineSpace(Degree(1), KnotVector([-width/2, width/2], [2,2]))
    S₂ = SplineSpace(Degree(1), KnotVector([-height/2, height/2], [2,2]))

    F = GeometricMapping(Nurbs, S₁ ⨷ S₂; codimension=2)

    F.weights .= 1
    F[1].coeffs[1,:] .= -width/2.
    F[1].coeffs[2,:] .= width/2
    F[2].coeffs[:,1] .= -height/2.
    F[2].coeffs[:,2] .= height/2
    
    return F
end

"""
    embedding(width::T, height::T, depth::T)

Return Cartesian background mapping.
"""
function embedding(width::T, height::T, depth::T) where {T<:Real}
    @assert depth > 0
    surf = embedding(width, height)

    S₃ = SplineSpace(Degree(1), KnotVector([-depth/2, depth/2], [2,2]))
    F = GeometricMapping(Nurbs, surf.space ⨷ S₃; codimension=3)

    F.weights .= 1
    F[1].coeffs[:,:,1] = F[1].coeffs[:,:,2] = surf[1].coeffs
    F[2].coeffs[:,:,1] = F[2].coeffs[:,:,2] = surf[2].coeffs
    F[3].coeffs[:,:,1] .= -depth/2.
    F[3].coeffs[:,:,2] .= depth/2
    
    return F
end