export system_matrix_integrand, weighted_system_matrix, get_system_matrix_quadrule


"""
    const TargetSpace{T} = UnivariateSplines.SplineSpace{T}

Alias for `UnivariateSplines.SplineSpace`.
"""
const TargetSpace{T} = UnivariateSplines.SplineSpace{T}

"""
    const TestSpace{T} = UnivariateSplines.SplineSpace{T}

Alias for `UnivariateSplines.SplineSpace`.
"""
const TestSpace{T} = UnivariateSplines.SplineSpace{T}

"""
    get_system_matrix_quadrule(S::TargetSpace, V::TestSpace)

Returns a univariate `PatchRule` for a pair of spaces.
"""
function get_system_matrix_quadrule(S::TargetSpace, V::TestSpace)
    # get knot vectors
    u = IncreasingVector(KnotVector(S))
    v = IncreasingVector(KnotVector(V))
    @assert u == v

    # construct global quadrature rule
    nq = ceil(Int, 0.5 * (Degree(S) + Degree(V) + 1))
    Q = PatchRule(u; npoints=nq, method=Legendre)
end

"""
    weighted_system_matrix(S::TargetSpace, V::TestSpace, data::AbstractVector, k::Int = 1, l::Int = 1)

Returns a univariate system matrix as in `UnivariateSplines.system_matrix` but
applies weighting to the test function defined by a `data` vector.

# Arguments:
- `S`: trial space
- `V`: test space
- `data`: vector with weights
- `k`: derivative order on trial functions
- `l`: derivarive order on test functions
"""
function weighted_system_matrix(S::TargetSpace, V::TestSpace, data::AbstractVector, k::Int=1, l::Int=1)
    # sanity check
    @assert (k in 1:Degree(S)+1) && (l in 1:Degree(V)+1)

    # obtain quadrule
    Q = get_system_matrix_quadrule(S, V)

    # check passed data
    @assert length(data) == length(Q.w)

    # compute B-spline basisfunctions of the test space
    Nu = ders_bspline_interpolation_matrix(Degree(S), KnotVector(S), Q.x, k)[k]

    # compute B-spline basisfunctions of the trial space
    Nv = ders_bspline_interpolation_matrix(Degree(V), KnotVector(V), Q.x, l)[l]

    return Nv' * ((data .* Q.w) .* Nu)
end

"""
    weighted_system_matrix(S::TargetSpace, V::TestSpace, data::Bspline, k::Int = 1, l::Int = 1)

Returns a univariate system matrix as in `UnivariateSplines.system_matrix` but
applies a spline weighting function to the test functions.

# Arguments:
- `S`: trial space
- `V`: test space
- `data`: univariate spline weighting function
- `k`: derivative order on trial functions
- `l`: derivarive order on test functions
"""
function weighted_system_matrix(S::TargetSpace, V::TestSpace, data::Bspline, k::Int=1, l::Int=1)
    # sanity check
    @assert (k in 1:Degree(S)+1) && (l in 1:Degree(V)+1)

    # obtain quadrule
    Q = get_system_matrix_quadrule(S, V)

    # compute weights from bspline weighting function
    @evaluate weights = data(Q.x)

    # check conforming length
    @assert length(weights) == length(Q.w)

    # compute B-spline basisfunctions of the test space
    Nu = ders_bspline_interpolation_matrix(Degree(S), KnotVector(S), Q.x, k)[k]

    # compute B-spline basisfunctions of the trial space
    Nv = ders_bspline_interpolation_matrix(Degree(V), KnotVector(V), Q.x, l)[l]

    return Nv' * ((weights .* Q.w) .* Nu)
end

"""
    system_matrix_integrand(S::TargetSpace{T}, V::TestSpace{T}, k::Int = 1, l::Int = 1; x::T)

Evaluates to the integrand of a system matrix `∫ s(ξ)v(ξ) dΩ` for `ξ = x` where `s ∈ S` and `v ∈ V`.

This is useful i.e. in the definition of boundary integrals on Cartesian grids.

# Arguments:
- `S`: a `UnivariateSplines.SplineSpace` as target space
- `V`: a `UnivariateSplines.SplineSpace` as test space
- `k`: derivative order on target space
- `l`: derivative order on test space
- `x`: evaluation point
"""
function system_matrix_integrand(S::TargetSpace{T}, V::TestSpace{T}, k::Int=1, l::Int=1; x::T) where {T <: Real}
    @assert (k in 1:Degree(S)+1) && (l in 1:Degree(V)+1)
    u = IncreasingVector(KnotVector(S))
    v = IncreasingVector(KnotVector(V))
    @assert u == v
    @assert u[1] <= x && x <= u[end]
    
    Nu = ders_bspline_interpolation_matrix(Degree(S), KnotVector(S), IncreasingVector([x]), k)[k]
    Nv = ders_bspline_interpolation_matrix(Degree(V), KnotVector(V), IncreasingVector([x]), l)[l]

    return Nv' * Nu
end