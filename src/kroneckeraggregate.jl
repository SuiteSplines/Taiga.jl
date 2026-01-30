export KroneckerProductAggregate, *, adjoint, reduce!, droptol!, get_factor

"""
    KroneckerProductAggregate{T,S<:KroneckerProduct{T}} <: LinearMap{T}

A collection of Kronecker product matrices that acts as a 
linear operator represented by a sum over that collection.

# Fields:
- `K::Vector{KroneckerProduct}`: collection of Kronecker products
- `size::Tuple{Int,Int}`: size of the operator
- `cache::Vector{T}`: matrix-vector product cache
- `isposdef::Bool`: operator is positive definite flag
- `ishermitian::Bool`: operator is hermitian flag
- `issymmetric::Bool`: operator is symmetric flag
"""
struct KroneckerProductAggregate{T,S<:KroneckerProduct{T}} <: LinearMap{T}
    K::Vector{S}
    size::Tuple{Int,Int}
    cache::Vector{T}
    isposdef::Bool
    ishermitian::Bool
    issymmetric::Bool

    @doc """
        KroneckerProductAggregate{T}(m::Int, n::Int; isposdef::Bool=false, ishermitian::Bool=false, issymmetric::Bool=false) where {T}

    Constructs an empty [`KroneckerProductAggregate`](@ref) of size `m × n` and
    properties defined by boolean keywords.
    """
    function KroneckerProductAggregate{T}(m::Int, n::Int; isposdef::Bool=false, ishermitian::Bool=false, issymmetric::Bool=false) where {T}
        cache = Vector{T}(undef, m)
        sz = (m,n)
        new{T,KroneckerProduct{T}}(Vector{KroneckerProduct{T}}(), sz, cache)
    end

    @doc """
        KroneckerProductAggregate{T}(m::Int, n::Int; isposdef::Bool=false, ishermitian::Bool=false, issymmetric::Bool=false) where {T}

    Constructs a [`KroneckerProductAggregate`](@ref) based on an arbitrary number of
    Kronecker products with equal size. Aggregate properties are defined by boolean keywords.
    """
    function KroneckerProductAggregate(K::Vararg{S,N}; isposdef::Bool=false, ishermitian::Bool=false, issymmetric::Bool=false) where {N,T,S<:KroneckerProduct{T}}
        sz = Base.size(K[1])
        for k in K
            @assert size(k) == sz "KroneckerProducts to be aggregated are not of the same product size"
        end
        cache = Vector{T}(undef, sz[1])
        new{T,S}(collect(K), sz, cache, isposdef, ishermitian, issymmetric)
    end
end

LinearAlgebra.isposdef(K::KroneckerProductAggregate) = K.isposdef
LinearAlgebra.issymmetric(K::KroneckerProductAggregate) = K.issymmetric
LinearAlgebra.ishermitian(K::KroneckerProductAggregate) = K.ishermitian
Base.isempty(K::KroneckerProductAggregate) = isempty(K.K)
Base.size(K::KroneckerProductAggregate) = K.size
Base.length(K::KroneckerProductAggregate) = length(K.K)
get_factor(K::KroneckerProductAggregate, r::Int, k::Int) = K.K[r].data[k]

"""
    Base.push!(K::KroneckerProductAggregate{T}, k::S) where {T,S<:KroneckerProduct{T}}

Adds a Kronecker product matrix to the collection of Kronecker products in `K`.
"""
function Base.push!(K::KroneckerProductAggregate{T}, k::S) where {T,S<:KroneckerProduct{T}}
    @assert Base.size(k) == K.size "Kronecker product to be aggregated is not of correct product size"
    push!(K.K, k)
    return K
end

"""
    *(K::KroneckerProductAggregate, v::S)

Compute `K.K[1] * v + K.K[2] * v + ... + K.K[end] * v`.

# Arguments:
- `K`: collection of Kronecker products
- `v`: vector to multiply with
"""
function Base.:*(K::KroneckerProductAggregate{T}, v::S) where {T,S<:AbstractVector}
    @assert size(v, 1) == K.size[2] "incompatible vector size"
    b = Vector{T}(undef, K.size[1])
    mul!(b, K, v)
    return b
end

function LinearMaps._unsafe_mul!(b::AbstractVector, K::KroneckerProductAggregate, v::S) where {S<:AbstractVector}
    b .= 0
    for k in K.K
        mul!(K.cache, k, v)    
        b .+= K.cache
    end
    return b
end

"""
    LinearAlgebra.adjoint(K::KroneckerProductAggregate)

Returns an adjoint `KroneckerProductAggregate`.
"""
function LinearAlgebra.adjoint(K::KroneckerProductAggregate)
    return KroneckerProductAggregate(map(transpose, K.K)...)
end

"""
    droptol!(K::S; rtol = √(eps(T)))

Drop contributions in `K = {K₁, K₂, ...}` for which the relative operator norm
`‖Kₖ‖₂ / || ‖Kₘₐₓ‖₂` is less then `rtol`, where for `Kₖ ∈ K` we find
`Kₘₐₓ = argmax ‖Kₖ‖₂`.
"""
function droptol!(K::S; rtol=√eps(T)) where {T,S<:KroneckerProductAggregate{T}}
    val = opnorm.(K.K)
    ref = maximum(val)
    dropmarker = [val[k]/ref < rtol for k in Base.OneTo(length(K.K))]
    deleteat!(K.K, dropmarker)
end