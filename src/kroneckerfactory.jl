export KroneckerFactory, reset!

"""
    KroneckerFactory{Dim, T, S <: SplineSpace{T}, K <: KroneckerProductAggregate{T}}

A factory that assembles Kronecker product system matrices. Acts as a functor
accepting optional weighting for test functions and collects Kronecker product
matrix contributions in a [`KroneckerProductAggregate`](@ref).

# Fields:
- `trialspace::TensorProduct{Dim, S}`: trial functions space
- `testspace::TensorProduct{Dim, S}`: test functions space
- `data::K`: Kronecker product contributions
"""
struct KroneckerFactory{Dim,T,S<:SplineSpace{T},K<:KroneckerProductAggregate{T}}
    trialspace::TensorProduct{Dim,S}
    testspace::TensorProduct{Dim,S}
    data::K
end

"""
    KroneckerFactory(trialspace::TensorProduct{Dim, S}, testspace::TensorProduct{Dim, S}) where {Dim,T,S<:SplineSpace{T}}

Constructs [`KroneckerFactory`](@ref).

# Arguments:
- `trialspace`: trial functions space
- `testspace`: test functions space
"""
function KroneckerFactory(trialspace::TensorProduct{Dim,S}, testspace::TensorProduct{Dim,S}) where {Dim,T,S<:SplineSpace{T}}
    m = dimension(testspace)
    n = dimension(trialspace)
    data = KroneckerProductAggregate{T}(m,n)
    KroneckerFactory(trialspace, testspace, data)
end

"""
    f::KroneckerFactory{Dim}(v::NTuple{N, Int}, w::NTuple{N, Int}; data = nothing, reset::Bool=false)

Assembles Kronecker product system matrices with test functions weighted by `data` (optional).
The Kronecker product contributions are collected in the `f.data` which is a 
[`KroneckerProductAggregate`](@ref).

For applicable data see [`weighted_system_matrix`](@ref).

# Arguments:
- `v`: tuple indicating trial function derivatives, see [`ι`](@ref)
- `w`: tuple indicating test function derivatives, see [`ι`](@ref)
- `data`: optional weighting of test functions
- `reset`: optional flag to reset the collection `f.data`
"""
function (f::KroneckerFactory{Dim})(v::NTuple{N,Int}, w::NTuple{N,Int}; data=nothing, α::T=1.0, reset::Bool=false) where {Dim,N,T<:Real}
    # reset collection if requested
    reset && reset!(f)

    # assemble contribution
    K = α * KroneckerProduct(f(v, w, data)...; reverse=true)

    # add contribution to collection
    push!(f.data, K)

    # return contribution
    return K
end

function (f::KroneckerFactory{Dim})(trialders::NTuple{N,Int}, testders::NTuple{N,Int}, data::Nothing) where {Dim,N}
    ntuple(k -> Matrix(f.testspace[k].C' * system_matrix(f.trialspace[k], f.testspace[k], trialders[k] + 1, testders[k] + 1) * f.trialspace[k].C), Dim)
end

function (f::KroneckerFactory{Dim})(trialders::NTuple{N,Int}, testders::NTuple{N,Int}, data) where {Dim,N}
    ntuple(k -> Matrix(f.testspace[k].C' * weighted_system_matrix(f.trialspace[k], f.testspace[k], data[k], trialders[k] + 1, testders[k] + 1) * f.trialspace[k].C), Dim)
end

"""
    reset!(f::KroneckerFactory)

Reset [`KroneckerProductAggregate`](@ref) collection in [`KroneckerFactory`](@ref).
"""
reset!(f::KroneckerFactory) = deleteat!(f.data.K, eachindex(f.data.K))