export ι, extreme_eigenvalues

"""
    ι(k::Int; val::Int=1, dim::Int=3)

Brief magic function that returns a tuple of integers with
length `dim` where `k`-th integer is equal to `val` and the rest is zero.

This is in particular useful to define tuples for test and trial
function derivatives in sum factorization loops, e.g.

Example:
```jldoctest
julia> u = ι(0, dim = 2);

julia> ∇u(k) = ι(k, dim = 2);

julia> u
(0, 0)

julia> ∇u(1)
(1, 0)

julia> ∇u(2)
(0, 1)

julia> ∇²ₖₖu(k) = ι(k, val=2, dim = 2);

julia> ∇²ₖₖu(1)
(2, 0)

julia> ∇²ₖₖu(2)
(0, 2)
```
"""
@inline ι(k::Int; val::Int=1, dim::Int=3) = ntuple(i -> val * Int(i == k), dim)


"""
    ι(k::Int, l::Int; val::NTuple{2,Int}=(1,1), dim::Int=3)

Brief magic function that returns a tuple of integers with
length `dim` where `k`-th integer is equal to `val[1]`,
`l`-th integer is equal to `val[2]` and the rest is zero.
If `k` and `l` coincide, the values are summed up.

This is in particular useful to define tuples for test and trial
function derivatives in sum factorization loops, e.g.

Example:
```jldoctest
julia> ∇²u(k, l) = ι(k, l, dim = 2);

julia> ∇²u(1, 1)
(2, 0)

julia> ∇²u(1, 2)
(1, 1)

julia> ∇²u(2, 1)
(1, 1)

julia> ∇²u(2, 2)
(0, 2)
```
"""
@inline ι(k::Int, l::Int; val::NTuple{2,Int}=(1, 1), dim::Int=3) = ntuple(i -> val[1] * Int(i == k) + val[2] * Int(i == l), dim)


"""
    LinearAlgebra.opnorm(A::K) where {Dim,T<:Real,K<:KroneckerProduct{T,2,Dim}}

Returns the operator norm `‖A‖₂` of a Kronecker product using Kronecker product
eigenvalue decomposition of `AᵀA` which is fast.
"""
function LinearAlgebra.opnorm(A::K) where {Dim,T<:Real,K<:KroneckerProduct{T,2,Dim}}
    λ² = eigen(A' * A).values
    sqrt(maximum(λ²))
end

# bug fix for AbstractMappings:
# https://gitlab.com/feather-ecosystem/core/abstractmappings/-/blob/master/src/evaluate.jl?ref_type=heads#L101
# type should not be constrained to EvaluationSet!
import AbstractMappings.evaluate_imp!
@inline function AbstractMappings.evaluate_imp!(op, z, x, f::Pairing)
    @evaluate y = f.mapping(x)
    evaluate_imp!(op, z, y, f.field)
    return z
end

"""
    extreme_eigenvalues(A::L; tol::T = 0.1, restarts::Int=200) where {T,L<:LinearMap}

Compute extreme eigenvalues of a linear map `A`. Uses `ArnoldiMethod`.
The tolerance `tol` can be helpful if `ArnoldiMethod` converges to more
then one maximum or minimum eigenvalue. Checks for complex eigenvalues
are performed.
"""
function extreme_eigenvalues(A::L; tol::T=10e-2, restarts::Int=200) where {T,L<:LinearMap}
    lm, lmlog = partialschur(A; nev=1, tol=tol, which=:LM, restarts=restarts)
    sr, srlog = partialschur(A; nev=1, tol=tol, which=:SR, restarts=restarts)

    @assert length(lm.eigenvalues) > 0
    @assert length(sr.eigenvalues) > 0

    length(lm.eigenvalues) > 1 && @warn "ArnoldiMethod converged to more than one largest eigenvalue!"
    length(sr.eigenvalues) > 1 && @warn "ArnoldiMethod converged to more than one smallest eigenvalue!"

    λmax, λmin = lm.eigenvalues[1], sr.eigenvalues[1]
    if abs(imag(λmax)) < 100eps(T) && abs(imag(λmin)) < 100eps(T)
        return real(λmax), real(λmin)
    end

    return λmax, λmin
end





## to be incorporated into Feather ##

galerkin_project!(s; onto) = map((sₖ,gₖ) -> galerkin_project!(sₖ, onto=gₖ), s, onto)
#function IgaBase.project!(spline::TensorProductBspline; onto, method::Type{<:AbstractProjection})
function galerkin_project!(spline::TensorProductBspline; onto)
    quadrule = standard_quadrature_rule(onto, spline)
    update!(spline.cache, quadrule.x)
    @evaluate y = onto(quadrule.x)
    #IgaBase.project_imp!(method, spline, quadrule, y)
    galerkin_project_imp!(GalerkinProjection, spline, quadrule, y)
    nothing
end
#@inline function IgaBase.project_imp!(::Type{GalerkinProjection}, spline::TensorProductBspline, quadrule, y)
function galerkin_project_imp!(::Type{GalerkinProjection}, spline::TensorProductBspline, quadrule, y)
    M = KroneckerProduct(Matrix ∘ system_matrix, spline.space, spline.space; reverse=true)
    N = KroneckerProduct(Matrix ∘ bspline_interpolation_matrix, spline.space, quadrule.x.data; reverse=true)
    b = N' * (quadrule.w .* view(y, :))
    spline.coeffs[:] = inv(M) * b
end

struct MixedRefinement{N,T} <: AbstractRefinement
    collection::NTuple{N,T}
    function MixedRefinement(R::Vararg{AbstractRefinement,N}) where {N}
        for r in R
            @assert parameterless_typeof(r) !== MixedRefinement
        end
        return new{N,AbstractRefinement}(R)
    end
end
Base.length(::MixedRefinement{N}) where {N} = N
Base.iterate(R::MixedRefinement) = Base.iterate(R.collection)
Base.iterate(R::MixedRefinement, i) = Base.iterate(R.collection, i)

struct NoRefinement <: AbstractRefinement end
function UnivariateSplines.refinement_operator(splinespace::SplineSpace, method::NoRefinement)
    return sparse(I, dimsplinespace(splinespace), dimsplinespace(splinespace)), splinespace
end

function IgaBase.refine_imp(spline::TensorProductBspline, method::MixedRefinement)
    # determine refinement operator and new spline spaces
    extraction_operators, univariate_spaces = unzip(map((s,m) -> refinement_operator(s, m), spline.space, method))
    C = KroneckerProduct(c -> Matrix(c), extraction_operators; reverse=true)
    space = TensorProduct(univariate_spaces...)

    # compute new coefficients
    coeffs = zeros(eltype(spline.coeffs), size(space))
    @kronecker! coeffs = C * spline.coeffs

    return TensorProductBspline(space, coeffs; ders=spline.ders, orientation=spline.orientation)
end
