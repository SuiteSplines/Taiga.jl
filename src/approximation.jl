export DataApproximationMethod, ModalSplines, Wachspress, CanonicalPolyadic
export approximate_patch_data

"""
    abstract type DataApproximationMethod

Concrete data approximatino methods derive from this abstract type.
"""
abstract type DataApproximationMethod end

"""
    CanonicalPolyadic <: DataApproximationMethod

Method to approximate patch data as separable functions
using Canonical polyadic decomposition.
"""
struct CanonicalPolyadic <: DataApproximationMethod end

"""
    NonnegativeCanonicalPolyadic <: DataApproximationMethod

Method to approximate patch data in `R₊` as positive
separable functions using the non-negative Canonical polyadic decomposition.
"""
struct NonnegativeCanonicalPolyadic <: DataApproximationMethod end

"""
    ModalSplines <: DataApproximationMethod

Method to approximate patch data as separable functions
using (ho)svd method and spline interpolation.
"""
struct ModalSplines <: DataApproximationMethod end

"""
    ModalSpline <: DataApproximationMethod

Method to approximate patch data at quadrature points
using Wachspress algorithms. Works only for data arrays
with positive entries.
"""
struct Wachspress <: DataApproximationMethod end


"""
    approximate_patch_data

Methods implementing patch data approximation algorithms.
"""
function approximate_patch_data end

"""
    approximate_patch_data(C::E, args; method::Type{<:DataApproximationMethod}, kwargs) where {S1,S2,E<:EvaluationSet{S1,S2}}

Approximate patch data using `method`. Calls implementation of `approximation_patch_data` for
`method` with positional arguments `args` and keyword arguments `kwargs`.

# Arguments:
- `C`: patch data as `IgaFormation.EvaluationSet`
- `args`: positional arguments to method
- `method`: approximation method
- `kwargs`: keyword arguments to method
"""
approximate_patch_data(C::E, args...; method::Type{<:DataApproximationMethod}, kwargs...) where {S1,S2,E<:EvaluationSet{S1,S2}} = approximate_patch_data(method, C, args...; kwargs...)

function approximate_patch_data(::Type{<:ModalSplines}, C::E, spaces::NTuple{2,SplineSpace{T}}; rank::Int=1) where {S1,S2,E<:EvaluationSet{S1,S2},T}
    # initialize arrays for svd modes and weighting splines
    modes = Array{NTuple{2,Vector{T}}}(undef, rank, S1, S2)
    splines = Array{NTuple{2,Bspline{T}}}(undef, rank, S1, S2)

    # compute svd of each data array (C.data[k,l])
    D = svd.(C.data)

    # process modes for each data array
    for l in Base.OneTo(S2)
        for k in Base.OneTo(S1)
            for r in Base.OneTo(rank)
                σ = sqrt(D[k, l].S[r])
                modes[r, k, l] = (σ * D[k, l].U[:, r], σ * D[k, l].V[:, r])
            end
        end
    end

    # define weighting splines by interpolating modes in each parametric dimension
    X = grevillepoints.(spaces)
    for l in Base.OneTo(S2)
        for k in Base.OneTo(S1)
            for r in Base.OneTo(rank)
                # first parametric direction
                B₁ = Bspline(spaces[1])
                UnivariateSplines.update!(B₁.cache, X[1])
                B₁.coeffs .= B₁.cache.basis \ modes[r, k, l][1]

                # second parametric direction
                B₂ = Bspline(spaces[2])
                UnivariateSplines.update!(B₂.cache, X[2])
                B₂.coeffs .= B₂.cache.basis \ modes[r, k, l][2]

                # set splines for r-th mode of (k,l)-th data array
                splines[r, k, l] = (B₁, B₂)
            end
        end
    end

    return splines
end

function approximate_patch_data(::Type{<:ModalSplines}, C::E, spaces::NTuple{3,SplineSpace{T}}; rank::Int=1) where {S1,S2,E<:EvaluationSet{S1,S2},T}
    # initialize arrays for svd modes and weighting splines
    modes = Array{NTuple{3,Vector{T}}}(undef, rank^3, S1, S2)
    splines = Array{NTuple{3,Bspline{T}}}(undef, rank^3, S1, S2)
    D = Array{ttensor,2}(undef, S1, S2)

    # compute hosvd of each data array (C.data[k,l])
    for k in eachindex(D)
        D[k] = hosvd(C.data[k].data, reqrank=repeat([rank], 3))
    end

    # paranoia mode
    core_size = Base.size(D[1].cten)
    nmodes = prod(core_size)
    for k in eachindex(D)
        @assert prod(Base.size(D[k].cten)) == nmodes
    end

    # process modes for each data array
    cind = CartesianIndices(core_size)
    for l in Base.OneTo(S2)
        for k in Base.OneTo(S1)
            for r in Base.OneTo(nmodes)
                σ = cbrt(D[k, l].cten[r])
                modes[r, k, l] = (
                    σ * D[k, l].fmat[1][:, cind[r][1]],
                    σ * D[k, l].fmat[2][:, cind[r][2]],
                    σ * D[k, l].fmat[3][:, cind[r][3]]
                )
            end
        end
    end

    # define weighting splines by interpolating modes in each parametric dimension
    X = grevillepoints.(spaces)
    for l in Base.OneTo(S2)
        for k in Base.OneTo(S1)
            for r in Base.OneTo(nmodes)
                # first parametric direction
                B₁ = Bspline(spaces[1])
                UnivariateSplines.update!(B₁.cache, X[1])
                B₁.coeffs .= B₁.cache.basis \ modes[r, k, l][1]

                # second parametric direction
                B₂ = Bspline(spaces[2])
                UnivariateSplines.update!(B₂.cache, X[2])
                B₂.coeffs .= B₂.cache.basis \ modes[r, k, l][2]

                # third parametric direction
                B₃ = Bspline(spaces[3])
                UnivariateSplines.update!(B₃.cache, X[3])
                B₃.coeffs .= B₃.cache.basis \ modes[r, k, l][3]

                # set splines for r-th mode of (k,l)-th data array
                splines[r, k, l] = (B₁, B₂, B₃)
            end
        end
    end

    return splines
end

"""
    approximate_patch_data(::Type{<:ModalSplines}, C::E; S::NTuple{Dim,SplineSpace{T}}, rank::Int=1) where {Dim,S1,S2,E<:EvaluationSet{S1,S2},T}

Returns an array of size `rank × S1 × S2` of Bspline tuples of length `Dim` for each data array in `C`.

# Arguments:
- `C`: patch data as `IgaFormation.EvaluationSet`
- `spaces`: tuple of univariate interpolation spline spaces
- `rank`: approximation rank
"""
function approximate_patch_data(::Type{<:ModalSplines}, C::E; spaces::NTuple{Dim,SplineSpace{T}}, rank::Int=1) where {Dim,S1,S2,E<:EvaluationSet{S1,S2},T}
    approximate_patch_data(ModalSplines, C, spaces; rank)
end

function approximate_patch_data(::Type{<:Wachspress}, C::E; niter::Int=2) where {E<:EvaluationSet{2,2}}
    S1, S2 = 2, 2
    T = eltype(C.data[1])
    D = Array{NTuple{2,Vector{T}}}(undef, S1, S2)
    m, n = Base.size(C)

    # compute approximations of diagonal data blocks
    for s1 in Base.OneTo(S1)
        A = ones(m)
        B = zeros(n)

        for j in Base.OneTo(n)
            B[j] = sqrt(maximum(view(C.data[s1, s1], :, j)) * minimum(view(C.data[s1, s1], :, j)))
        end

        for k in Base.OneTo(niter)
            for i in Base.OneTo(m)
                A[i] = sqrt(maximum(view(C.data[s1, s1],i, :) ./ B) * minimum(view(C.data[s1, s1], i, :) ./ B))
            end

            for j in Base.OneTo(n)
                B[j] = sqrt(maximum(view(C.data[s1, s1], :, j) ./ A) * minimum(view(C.data[s1, s1], :, j) ./ A))
            end
        end

        D[s1, s1] = (A, B)
    end

    # set off-diagonal data block approximations to zero
    for s2 in Base.OneTo(S2)
        for s1 in Base.OneTo(S1)
            if s1 != s2
                D[s1,s2] = (zeros(m), zeros(n))
            end
        end
    end

    return D
end

function approximate_patch_data(::Type{<:Wachspress}, C::E; niter::Int=2) where {E<:EvaluationSet{3,3}}
    S1, S2 = 3, 3
    T = eltype(C.data[1])
    D = Array{NTuple{3,Vector{T}}}(undef, S1, S2)

    # allocate memory
    N = 3
    C = [ C.data[k,k] for k in 1:N ]
    n = size(C[1])
    μ = [ ones(n) for n in size(C[1]) ]
    ω = [ ones(n) for n in size(C[1]) ]
    B = [ similar(C[1]) for k in 1:N ]

    # compute approximations of diagonal data blocks
    for iter in 1:niter
        for k in 1:N
            B[k] .= C[k]
            view(B[k], :) ./= KroneckerProduct(ntuple(ind -> (ind == k) ? ones(n[ind]) : μ[ind], N)...; reverse=true)

            for j = 1:n[k]
                indices = ntuple(ind -> (ind == k) ? j : Colon(), N)
                m = minimum(view(B[k], indices...))
                M = maximum(view(B[k], indices...))
                ω[k][j] = sqrt(m*M)
            end
        end

        W = [ similar(C[1]) for k in 1:N, l in 1:N ]
        for k in 1:N
            for l in 1:N
                W[k,l] .= C[k] 
                if l != k
                    view(W[k,l], :) .*= KroneckerProduct(ntuple(ind -> (ind == l) ? μ[l] : ones(n[ind]), N)...; reverse=true)
                    view(W[k,l], :) ./= KroneckerProduct(ntuple(ind -> (ind == k) ? ω[k] : μ[ind], N)...; reverse=true)
                end
            end

            P = similar(C[1])
            for ind in eachindex(P)
                vals = [ W[k,l][ind] for l in setdiff(1:N, k) ]
                P[ind] = minimum(vals)
            end
            R = similar(C[1])
            for ind in eachindex(R)
                vals = [ W[k,l][ind] for l in setdiff(1:N, k) ]
                R[ind] = maximum(vals)
            end

            for j = 1:n[k]
                indices = ntuple(ind -> (ind == k) ? j : Colon(), N)
                m = minimum(view(P, indices...))
                M = maximum(view(R, indices...))
                μ[k][j] = sqrt(m*M)
            end
        end
    end

    # set diagonal data block approximations
    D[1,1] = (ω[1], μ[2], μ[3])
    D[2,2] = (μ[1], ω[2], μ[3])
    D[3,3] = (μ[1], μ[2], ω[3])

    # set off-diagonal data block approximations to zero
    m, n, o = Base.size(C[1])
    for s2 in Base.OneTo(S2)
        for s1 in Base.OneTo(S1)
            if s1 != s2
                D[s1,s2] = (zeros(m), zeros(n), zeros(o))
            end
        end
    end

    return D
end

function approximate_patch_data(::Type{<:CanonicalPolyadic}, D::Array{ktensor,2}, ::Val{3})
    # initialize arrays for cp modes
    S1, S2 = size(D)
    rank = length(D[1].lambda)
    T = eltype(D[1].fmat[1])
    modes = Array{NTuple{3,Vector{T}}}(undef, rank, S1, S2)

    # process modes for each data array
    for l in Base.OneTo(S2)
        for k in Base.OneTo(S1)
            for r in Base.OneTo(rank)
                σ = cbrt(D[k, l].lambda[r])
                modes[r, k, l] = (
                    σ * D[k, l].fmat[1][:, r],
                    σ * D[k, l].fmat[2][:, r],
                    σ * D[k, l].fmat[3][:, r]
                )
            end
        end
    end

    # return processed modes
    return modes
end

function approximate_patch_data(::Type{<:CanonicalPolyadic}, D::Array{ktensor,2}, ::Val{2})
    # initialize arrays for cp modes
    S1, S2 = size(D)
    rank = length(D[1].lambda)
    T = eltype(D[1].fmat[1])
    modes = Array{NTuple{2,Vector{T}}}(undef, rank, S1, S2)

    # process modes for each data array
    for l in Base.OneTo(S2)
        for k in Base.OneTo(S1)
            for r in Base.OneTo(rank)
                σ = sqrt(D[k, l].lambda[r])
                modes[r, k, l] = (
                    σ * D[k, l].fmat[1][:, r],
                    σ * D[k, l].fmat[2][:, r],
                )
            end
        end
    end

    # return processed modes
    return modes
end

"""
    approximate_patch_data(::Type{<:CanonicalPolyadic}, C::E; tol::T=10e-1, rank::Int=1, ntries::Int=10) where {Dim,S1,S2,E<:EvaluationSet{S1,S2},T}

Returns an array of size `rank × S1 × S2` of Vector tuples of length `Dim` for each data array in `C`.

# Arguments:
- `C`: patch data as `IgaFormation.EvaluationSet`
- `tol`: cp tolerance
- `rank`: number of cp modes
- `ntries`: number of attempts to compute the CP decomposition per block
"""
function approximate_patch_data(::Type{<:CanonicalPolyadic}, C::E; tol::T=10e-1, rank::Int=1, ntries::Int=10) where {T,S1,S2,E<:EvaluationSet{S1,S2}}
    # allocate array for canonical polyadic decompositions
    D = Array{ktensor,2}(undef, S1, S2)

    # compute cp decompositions of each data array (C.data[k,l])
    for k in eachindex(D)
        for l in 1:ntries
            try
                D[k] = cp_als(C.data[k].data, rank, tol=tol)
                (l > 1) && @warn "cp_als(C.data[$k].data, $rank, tol=$tol) needed $l attempts!"
                break
                # todo: shouldn't this throw error if l > ntries?!
            catch e
                (l == ntries) && throw(e)

                # if data is all zero cp_als will throw ArgumentError
                # we can treat that as zero tensor of appropriate size
                if e isa ArgumentError
                    if all(C.data[k].data .≈ 0)
                        @warn "cp_als(C.data[$k].data, $rank, tol=$tol) is all zero!"
                        D[k] = ktensor(zeros(rank), collect(map(sz -> zeros(sz,rank), size(C.data[k].data))))
                        break
                    end
                end
            end
        end
    end

    # get number of univariate directions
    nargs = length(D[1].fmat)

    # call appropriate mode processor respective to nargs
    approximate_patch_data(CanonicalPolyadic, D, Val(nargs))
end

#function als_rank_q_approx_inv(K::KroneckerProductAggregate{T}; tol::T=10e-3, niter::Int=2, q::Int=1, C0::NTuple{N,Matrix{T}}=ntuple(k -> rand(size(K.K[1].data[1],1), size(K.K[1].data[1],1)),q)) where {T,N}
#    # Yannis Voet, Preconditioning techniques for generalized Sylvester matrix equations (2024)
#    # https://arxiv.org/abs/2307.07884
#    @assert N == q
#
#    r = LinearAlgebra.rank(K)
#    n = size(K.K[1].data[1], 1)
#    m = size(K.K[1].data[2], 1)
#
#    C = BlockArray{T}(zeros(q*n,n), fill(n,q), [n])
#    for k in Base.OneTo(blocksize(C,1))
#        view(C, Block(k)) .= C0[k]
#    end
#    D = BlockArray{T}(rand(q*m,m), fill(m,q), [m])
#
#    A = r::Int -> get_factor(K, r, 1)
#    B = r::Int -> get_factor(K, r, 2)
#
#    AtA = BlockArray{T}(zeros(q*n,q*n), fill(n,q), fill(n,q))
#    BtB = BlockArray{T}(zeros(q*m,q*m), fill(m,q), fill(m,q))
#
#    AtI = BlockArray{T}(zeros(q*n,n), fill(n,q), [n])
#    BtI = BlockArray{T}(zeros(q*m,m), fill(m,q), [m])
#
#    αA = zeros(n,n)
#    βB = zeros(m,m)
#
#    β = BlockArray{T}(zeros(q*r,q*r), fill(r,q), fill(r,q))
#    δ = BlockArray{T}(zeros(q*r), fill(r,q))
#
#    α = BlockArray{T}(zeros(q*r,q*r), fill(r,q), fill(r,q))
#    γ = BlockArray{T}(zeros(q*r), fill(r,q))
#
#    j = 0
#    res = 10000
#    while sqrt(res) > tol && j < niter
#        for t in Base.OneTo(q)
#            for s in Base.OneTo(q)
#                for l in Base.OneTo(r)
#                    for k in Base.OneTo(r)
#                        view(β, Block(s,t))[k,l] = tr(A(k)' * A(l) * view(C, Block(s)) * view(C, Block(t))') # didn't use symmetry β = βᵀ!
#                    end
#                end
#            end
#        end
#
#        for s in Base.OneTo(q)
#            for k in Base.OneTo(r)
#                view(δ, Block(s))[k] = tr(A(k)' * view(C, Block(s)))
#            end
#        end
#
#        for t in Base.OneTo(q)
#            for s in Base.OneTo(q)
#                view(BtB, Block(s,t)) .= 0
#                for k in Base.OneTo(r)
#                    βB .= 0
#                    for l in Base.OneTo(r)
#                        βB += view(β, Block(s,t))[k,l] * B(l)
#                    end
#                    view(BtB, Block(s,t)) .+= βB' * B(k)
#                end
#            end
#        end
#
#
#        for s in Base.OneTo(q)
#            view(BtI, Block(s)) .= 0
#            for k in Base.OneTo(r)
#                view(BtI, Block(s)) .+= view(δ, Block(s))[k] * B(k)'
#            end
#        end
#
#        D .= Matrix(BtB) \ Matrix(BtI)
#
#        for t in Base.OneTo(q)
#            for s in Base.OneTo(q)
#                for l in Base.OneTo(r)
#                    for k in Base.OneTo(r)
#                        view(α, Block(s,t))[k,l] = tr(B(k)' * B(l) * view(D, Block(s)) * view(D, Block(t))') # didn't use symmetry β = βᵀ!
#                    end
#                end
#            end
#        end
#
#        for s in Base.OneTo(q)
#            for k in Base.OneTo(r)
#                view(γ, Block(s))[k] = tr(B(k)' * view(D, Block(s)))
#            end
#        end
#
#        for t in Base.OneTo(q)
#            for s in Base.OneTo(q)
#                view(AtA, Block(s,t)) .= 0
#                for k in Base.OneTo(r)
#                    αA .= 0
#                    for l in Base.OneTo(r)
#                        αA .+= view(α, Block(s,t))[k,l] * A(l)
#                    end
#                    view(AtA, Block(s,t)) .+= αA' * A(k)
#            
#                end
#            end
#        end
#
#        for s in Base.OneTo(q)
#            view(AtI, Block(s)) .= 0
#            for k in Base.OneTo(r)
#                view(AtI, Block(s)) .+= view(γ, Block(s))[k] * A(k)'
#            end
#        end
#
#        C .= Matrix(AtA) \ Matrix(AtI)
#
#        res = n*m - 2 * sum(γ .* δ) + sum(α .* β)
#        j += 1
#    end
#
#    return C.blocks, D.blocks
#end