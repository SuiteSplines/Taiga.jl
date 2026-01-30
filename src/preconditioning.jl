export Model
export Preconditioner, FastDiagonalization, InnerCG, InnerPCG, BlockDiagonalPreconditioner
export ApproximateKroneckerInverse
export inner_solver_niters, inner_solver_residuals, inner_solver_convergence
export reset_inner_solver_history!
export HyperPowerPreconditioner, hyperpower_initial_preconditioner
export hyperpower_eigenvalues_after_n_iterations, hyperpower_extreme_eigenvalues

"""
    abstract type Preconditioner{Dim,T}

Concrete preconditioners derive from this abstract type.
"""
abstract type Preconditioner{Dim,T} <: LinearMap{T} end

function Base.show(io::IO, P::T) where {T<:Preconditioner}
    sz = size(P)
    print(io, "Preconditioner of type $T with size $sz")
end

Base.size(::P) where {P<:Preconditioner} = error("$P does not implement Base.size()!")
LinearAlgebra.issymmetric(::P) where {P<:Preconditioner} = error("$P does not implement LinearAlgebra.issymmetric()!")
LinearAlgebra.ishermitian(::P) where {P<:Preconditioner}  = error("$P does not implement LinearAlgebra.ishermitian()!")
LinearAlgebra.isposdef(::P) where {P<:Preconditioner} = error("$P does not implement LinearAlgebra.isposdef()!")


"""
    FastDiagonalization{Dim,T,K<:KroneckerProduct{T}}

Fast diagonalization preconditioner.

# Fields:
- `Λ⁻¹::Diagonal{T, Vector{T}}`: diagonal matrix of reciprocal of positive eigenvalues
- `U::K`: Kronecker product eigenvectors
- `size::NTuple{2, Int}`: operator size
- `cache::Vector{T}`: intermediate product cache
"""
struct FastDiagonalization{Dim,T,K<:KroneckerProduct{T}} <: Preconditioner{Dim,T}
    Λ⁻¹::Diagonal{T,Vector{T}}
    U::K
    size::NTuple{2,Int}
    cache::Vector{T}
    function FastDiagonalization(Λ⁻¹::Diagonal{T,Vector{T}}, U::K) where {Dim,T,K<:KroneckerProduct{T, 2, Dim}}
        # check Λ⁻¹ ≻ 0 (positive eigenvalues)
        @assert isposdef(Λ⁻¹)

        # check conforming size
        @assert size(Λ⁻¹) == size(U)

        # operator size
        sz = size(U)

        # intermediate product cache
        cache = Vector{T}(undef, sz[1])

        return new{Dim,T,typeof(U)}(Λ⁻¹, U, sz, cache)
    end
end

Base.size(P::FastDiagonalization) = P.size
LinearAlgebra.issymmetric(::FastDiagonalization) = true
LinearAlgebra.ishermitian(::FastDiagonalization) = true
LinearAlgebra.isposdef(::FastDiagonalization) = true

function LinearMaps._unsafe_mul!(b::AbstractVecOrMat, P::FastDiagonalization, v::AbstractVector)
    mul!(b, P.U', v)
    mul!(P.cache, P.Λ⁻¹, b)
    mul!(b, P.U, P.cache)
end

"""
    InnerCG{Dim, T, L <: LinearOperatorApproximation} <: Preconditioner{Dim,T}

Preconditioner using inner CG solver. In applications the linear operator approximation
`A` can be cheaply applied to a vector. The parameter `η ∈ [0,1)` controls the
tolerance of the inner solve. Decreasing `η` increases the number of inner iterations.

The history can be reset using [`reset_inner_solver_history!`](@ref).

# Fields:
- `A::L`: approximation of a linear operator to be preconditioned
- `p::Vector{T}`: cache vector
- `Ap::Vector{T}`: cache vector
- `r::Vector{T}`: cache vector
- `η::T`: factor for convergence criterion (`η ∈ [0,1)`)
- `η̂::T`: factor for convergence criterion respective `cond(A)`
- `itmax::Int`: maximum number of inner iterations
- `convergence::Vector{Bool}`: vector with history of convergence
- `residuals::Vector{T}`: vector with history of residuals
- `niters::Vector{Int}`: vector with history of number of iterations
- `history::Bool`: boolean flag for history keeping
"""
struct InnerCG{Dim,T,L<:LinearOperatorApproximation} <: Preconditioner{Dim,T}
    A::L
    p::Vector{T}
    Ap::Vector{T}
    r::Vector{T}
    η::T
    η̂::T
    itmax::Int
    convergence::Vector{Bool}
    residuals::Vector{T}
    niters::Vector{Int}
    history::Bool

    @doc """
        InnerCG(A::L; history=false, itmax::Int=200, η::T=10e-5, eigtol::T=10e-2) where {Dim,T,L<:LinearOperatorApproximation{Dim,T}}

    Construct [`InnerCG`](@ref) preconditioner. The positive definiteness is
    checked by [`extreme_eigenvalues`](@ref). The `eigtol` keyword controls the tolerance
    for the extreme eigenvalues computation.

    # Arguments:
    - `A::L`: approximatino of a linear operator to be preconditioned
    - `history::Bool`: boolean flag for history keeping
    - `η::T`: factor for convergence criterion (`η ∈ [0,1)`)
    - `skip_checks::Bool`: boolean flag for skipping posdef checks (default: `false`)
    - `itmax::Int`: maximum number of inner iterations
    - `eigtol::T`: tolerance for `extreme_eigenvalues`](@ref) computation
    - `eigrestarts::Int`: number of restarts for `extreme_eigenvalues`](@ref) computation
    """
    function InnerCG(A::L; history::Bool=false, itmax::Int=200, η::T=10e-5, skip_checks::Bool=false, eigtol::T=10e-2, eigrestarts::Int=200) where {Dim,T,L<:LinearOperatorApproximation{Dim,T}}
        @assert η ≥ 0.0 && η < 1.0
        n = size(A, 2)

        if !skip_checks
            λmax, λmin = extreme_eigenvalues(A; tol=eigtol, restarts=eigrestarts)
            @assert isreal(λmax) && isreal(λmin)
            (λmax ≤ 0 || λmin ≤ 0) && error("Approximation A::$L is not positive definite!")
            η̂ = η / sqrt(λmax / λmin)
        else
            η̂ = η
        end

        p = zeros(n)
        Ap = zeros(n)
        r = zeros(n)
        convergence = Vector{Bool}()
        residuals = Vector{T}()
        niters = Vector{Int}()
        new{Dim,T,L}(A, p, Ap, r, η, η̂, itmax, convergence, residuals, niters, history)
    end
end

Base.size(P::InnerCG) = size(P.A)
LinearAlgebra.issymmetric(::InnerCG) = true
LinearAlgebra.ishermitian(::InnerCG) = true
LinearAlgebra.isposdef(::InnerCG) = true

function LinearMaps._unsafe_mul!(x::AbstractVecOrMat, P::InnerCG{Dim,T}, rₙ::AbstractVector) where {Dim,T}
    r, A, p, Ap = P.r, P.A, P.p, P.Ap
    η̂, itmax = P.η̂, P.itmax

    niter = 0
    converged = false
    n = size(A, 2)
    ρ = sqrt(kdotr(n, rₙ, rₙ))

    # initialize solution vector
    x .= 0

    # compute residual
    mul!(r, A, x)
    kaxpby!(n, one(T), rₙ, -one(T), r)

    # compute squared residual norm
    γ = kdotr(n, r, r)

    # return if x0 good enough
    if (sqrt(γ) ≤ η̂ * ρ)
        P.history && push!(P.niters, 0)
        P.history && push!(P.residuals, sqrt(γ))
        P.history && push!(P.convergence, true)
        return x
    end

    # set initial direction
    copyto!(p, r)

    # iterative solve
    while (niter < itmax)
        mul!(Ap, A, p)
        α = γ / kdotr(n, p, Ap)
        kaxpy!(n, α, p, x)
        β = 1 / γ
        kaxpy!(n, -α, Ap, r)
        γ = kdotr(n, r, r)

        niter += 1
        (sqrt(γ) ≤ η̂ * ρ) && (converged = true; break)

        β *= γ
        kaxpby!(n, one(T), r, β, p)
    end

    !converged && @warn "InnerCG solve did not converge!"
    P.history && push!(P.niters, niter)
    P.history && push!(P.residuals, sqrt(γ))
    P.history && push!(P.convergence, converged)
    return x
end

"""
    InnerPCG{Dim, T, L <: LinearOperatorApproximation{Dim, T}, P <: Preconditioner{Dim, T}}

Preconditioner using inner PCG solver. In applications the linear operator approximation
`A` can be cheaply applied to a vector. The preconditioner `P` preconditions the inner solve.
The parameter `η ∈ [0,1)` controls the tolerance of the inner solve.
Decreasing `η` increases the number of inner iterations.

The history can be reset using [`reset_inner_solver_history!`](@ref).

# Fields:
- `A::L`: approximation of a linear operator to be preconditioned
- `M::P`: preconditioner for the inner solve
- `p::Vector{T}`: cache vector
- `Ap::Vector{T}`: cache vector
- `r::Vector{T}`: cache vector
- `z::Vector{T}`: cache vector
- `η::T`: factor for convergence criterion (`η ∈ [0,1)`)
- `η̂::T`: factor for convergence criterion respective `cond(A)`
- `itmax::Int`: maximum number of inner iterations
- `convergence::Vector{Bool}`: vector with history of convergence
- `residuals::Vector{T}`: vector with history of residuals
- `niters::Vector{Int}`: vector with history of number of iterations
- `history::Bool`: boolean flag for history keeping
"""
struct InnerPCG{Dim,T,L<:LinearOperatorApproximation{Dim,T},P<:Preconditioner{Dim,T}} <: Preconditioner{Dim,T}
    A::L
    M::P
    p::Vector{T}
    Ap::Vector{T}
    r::Vector{T}
    z::Vector{T}
    η::T
    η̂::T
    itmax::Int
    convergence::Vector{Bool}
    residuals::Vector{T}
    niters::Vector{Int}
    history::Bool

    @doc """
        InnerPCG(A::L, P; history=false, itmax::Int=200, η::T=10e-5, eigtol::T=10e-2) where {Dim,T,L<:LinearOperatorApproximation{Dim,T}}

    Construct [`InnerCG`](@ref) preconditioner. The positive definiteness is
    checked by [`extreme_eigenvalues`](@ref). The `eigtol` keyword controls the tolerance
    for the extreme eigenvalues computation.

    # Arguments:
    - `A::L`: approximatino of a linear operator to be preconditioned
    - `history::Bool`: boolean flag for history keeping
    - `η::T`: factor for convergence criterion (`η ∈ [0,1)`)
    - `skip_checks::Bool`: boolean flag for skipping posdef checks (default: `false`)
    - `itmax::Int`: maximum number of inner iterations
    - `eigtol::T`: tolerance for `extreme_eigenvalues`](@ref) computation
    - `eigrestarts::Int`: number of restarts for `extreme_eigenvalues`](@ref) computation
    """
    function InnerPCG(A::L, M::P; history::Bool=false, itmax::Int=200, η::T=10e-5, skip_checks::Bool=false, eigtol::T=10e-2, eigrestarts::Int=200, eigreal::Bool=true) where {Dim,T,L<:LinearOperatorApproximation{Dim,T},P<:Preconditioner{Dim,T}}
        @assert η ≥ 0.0 && η < 1.0
        n = size(A, 2)

        if !skip_checks
            λmax, λmin = extreme_eigenvalues(M*A; tol=eigtol, restarts=eigrestarts)
            if !eigreal
                # if unsure, set lower eigtol and eigreal=false
                @assert isreal(λmax) && isreal(λmin)
            else
                λmax, λmin = real(λmax), real(λmin)
            end
            (λmax ≤ 0 || λmin ≤ 0) && error("Approximation A::$L is not positive definite!")
            η̂ = η / sqrt(λmax / λmin)
        else
            η̂ = η
        end

        p = zeros(n)
        Ap = zeros(n)
        r = zeros(n)
        z = zeros(n)
        convergence = Vector{Bool}()
        residuals = Vector{T}()
        niters = Vector{Int}()
        new{Dim,T,L,P}(A, M, p, Ap, r, z, η, η̂, itmax, convergence, residuals, niters, history)
    end
end

Base.size(P::InnerPCG) = size(P.A)
LinearAlgebra.issymmetric(::InnerPCG) = true
LinearAlgebra.ishermitian(::InnerPCG) = true
LinearAlgebra.isposdef(::InnerPCG) = true

function LinearMaps._unsafe_mul!(x::AbstractVecOrMat, P::InnerPCG{Dim,T}, rₙ::AbstractVector) where {Dim,T}
    r, A, M, p, Ap, z = P.r, P.A, P.M, P.p, P.Ap, P.z
    η̂, itmax = P.η̂, P.itmax

    niter = 0
    converged = false
    n = size(A, 2)
    ρ = sqrt(kdotr(n, rₙ, rₙ))

    # initialize solution vector
    x .= 0

    # compute residual
    mul!(r, A, x)
    kaxpby!(n, one(T), rₙ, -one(T), r)

    # compute squared residual norm
    γ = kdotr(n, r, r)

    # return if x0 good enough
    if (sqrt(γ) ≤ η̂ * ρ)
        P.history && push!(P.niters, 0)
        P.history && push!(P.residuals, sqrt(γ))
        P.history && push!(P.convergence, true)
        return x
    end

    # apply preconditioner to residual
    mul!(z, M, r)

    # set initial direction
    copyto!(p, z)

    # iterative solve
    while (niter < itmax)
        mul!(Ap, A, p)
        δ = kdotr(n, r, z)
        α = δ / kdotr(n, p, Ap)
        kaxpy!(n, α, p, x)

        niter += 1
        (sqrt(γ) ≤ η̂ * ρ) && (converged = true; break)
        
        β = 1 / δ
        kaxpy!(n, -α, Ap, r)
        γ = kdotr(n, r, r)
        mul!(z, M, r)
        δ = kdotr(n, r, z)
        β *= δ
        kaxpby!(n, one(T), z, β, p)
    end

    !converged && @warn "InnerCG solve did not converge!"
    P.history && push!(P.niters, niter)
    P.history && push!(P.residuals, sqrt(γ))
    P.history && push!(P.convergence, converged)
    return x
end

"""
    reset_inner_solver_niter_history!(P::S) where {S<:Union{InnerCG, InnerPCG}}

Reset history in [`InnerCG`](@ref) and [`InnerPCG`](@ref) preconditioner.
"""
function reset_inner_solver_history!(P::S) where {S<:Union{InnerCG, InnerPCG}}
    @assert P.history == true
    deleteat!(P.niters, eachindex(P.niters))
    deleteat!(P.residuals, eachindex(P.residuals))
    deleteat!(P.convergence, eachindex(P.convergence))
    nothing
end

"""
    inner_solver_niter(P::S) where {S<:Union{InnerCG, InnerPCG}}

Returns a vector with numbers of iterations in last applications of [`InnerCG`](@ref) and
[`InnerPCG`](@ref) preconditioner.

See [`reset_inner_solver_history!`](@ref) to reset history.
"""
function inner_solver_niters(P::S) where {S<:Union{InnerCG, InnerPCG}}
    @assert P.history == true
    return P.niters
end

"""
    inner_solver_residuals(P::S) where {S<:Union{InnerCG, InnerPCG}}

Returns a vector with residuals in last applications of [`InnerPCG`](@ref) preconditioner.
See [`reset_inner_solver_history!`](@ref) to reset history.
"""
function inner_solver_residuals(P::S) where {S<:Union{InnerCG, InnerPCG}}
    @assert P.history == true
    return P.residuals
end

"""
    inner_solver_convergence(P::S) where {S<:Union{InnerCG, InnerPCG}}

Returns a vector booleans indicating convergence
in last applications of [`InnerPCG`](@ref) preconditioner.

See [`reset_inner_solver_history!`](@ref) to reset history.
"""
function inner_solver_convergence(P::S) where {S<:Union{InnerCG, InnerPCG}}
    @assert P.history == true
    return P.convergence
end

"""
    HyperPowerPreconditioner{Dim, T, A <: LinearOperatorApproximation{Dim, T}, B <: Preconditioner{Dim, T}}

Preconditioner based on the [Ben-Israel-Cohen iteration](http://benisrael.net/COHEN-BI-ITER-GI.pdf).

```
Pₖ₊₁⁻¹ = 2 Pₖ⁻¹ - Pₖ⁻¹ A Pₖ⁻¹     (Pₖ⁻¹ → A⁻¹   for   k → ∞)
```

Requires a linear operator `A` and an initial preconditioner `P`. Computes one update of the
Ben-Israel-Cohen iteration. Convergence to `A⁻¹` and positive definiteness are guaranteed
as long as `σ(P⁻¹A) ⊂ (0,2)`.

For recursive application, i.e. `n` iterations, use the convenience constructor

```
HyperPowerPreconditioner(L::A, P::B, n::Int) where {Dim,T,A<:LinearOperatorApproximation{Dim,T},B<:Preconditioner{Dim,T}}
```

where `P` is the initial preconditioner and `n` is the number of updates.

# Fields:
- `A::A`: linear operator of which the inverse is approximated
- `P::B`: initial preconditioner
- `v₁::Vector{T}`: vector cache
- `v₂::Vector{T}`: vector cache
"""
struct HyperPowerPreconditioner{Dim,T,A<:LinearOperatorApproximation{Dim,T},B<:Preconditioner{Dim,T}} <: Preconditioner{Dim,T}
    A::A
    P::B
    v₁::Vector{T}
    v₂::Vector{T}
    function HyperPowerPreconditioner(L::A, P::B) where {Dim,T,A<:LinearOperatorApproximation{Dim,T},B<:Preconditioner{Dim,T}}
        v₁ = zeros(size(P, 1))
        v₂ = zeros(size(P, 1))
        new{Dim,T,A,B}(L, P, v₁, v₂)
    end
end

function HyperPowerPreconditioner(L::A, P::B, n::Int) where {Dim,T,A<:LinearOperatorApproximation{Dim,T},B<:Preconditioner{Dim,T}}
    @assert n > 0
    X = HyperPowerPreconditioner(L, P)
    for k in 2:n
        X = HyperPowerPreconditioner(L, X)
    end
    return X
end

function LinearMaps._unsafe_mul!(b::AbstractVecOrMat, P::HyperPowerPreconditioner, v::AbstractVector)
    mul!(P.v₁, P.P, v)
    b .= 2P.v₁

    mul!(P.v₂, P.A, P.v₁)
    mul!(P.v₁, P.P, P.v₂)
    b .-= P.v₁
end

function Base.show(io::IO, P::HyperPowerPreconditioner{Dim,T,A,B}) where {Dim,T,A<:LinearOperatorApproximation{Dim,T},B<:HyperPowerPreconditioner{Dim,T}}
    sz, X = size(P), P.P
    n = 1
    while typeof(X) <: HyperPowerPreconditioner
        X = X.P
        n += 1
    end
    B0 = typeof(X)
    print(io,       "Preconditioner of type HyperPowerPreconditioner{$Dim,$T,")
    printstyled(io, "$A", color=:green, bold=true)
    println(io,     ",B<:HyperPowerPreconditioner{Dim,T}} and size $sz.")
    print(io,       "The initial preconditioner ")
    printstyled(io, "$B0"; color=:green, bold=true)
    print(io,       " was updated ")
    printstyled(io, "$n"; color=:green, bold=true)
    print(io,       " times.")
end

"""
    hyperpower_initial_preconditioner(P::H)

Returns the initial preconditioner that was used in the construction of `P`, e.g. [`FastDiagonalization`](@ref).
"""
function hyperpower_initial_preconditioner(P::H) where {Dim,T,H<:HyperPowerPreconditioner{Dim,T}}
    X = P.P
    while typeof(X) <: HyperPowerPreconditioner
        X = X.P
    end
    X
end

"""
    hyperpower_eigenvalues_after_n_iterations(λ::T; n::Int)

For a set of eigenvalues `λ` return the corresponding eigenvalues after `n`
further updates of the Ben-Israel-Cohen iteration.

Transformation rule applies for `λ` after first update. The upper bound
for `λ` after first update is 1.

Example:
```julia
julia> λ₀ = hyperpower_extreme_eigenvalues(P; n=0, tol=10e-5)
(0.023145184193109333, 1.9749588237991926)

julia> λ₁ = hyperpower_extreme_eigenvalues(P; n=1, tol=10e-5)
(0.045754651575638426, 0.9999899575917253)

julia> hyperpower_extreme_eigenvalues(P; n=5, tol=10e-5)
(0.5273271247385951, 0.9999962548752878)

julia> hyperpower_eigenvalues_after_n_iterations(λ₁; n=4)
(0.5273268941262143, 1.0)

julia> hyperpower_eigenvalues_after_n_iterations(λ₀; n=1)
(0.0457546604483684, 0.049592390484542115)
```
"""
function hyperpower_eigenvalues_after_n_iterations(λ::T; n::Int) where {T}
    n == 0 && return λ
    hyperpower_eigenvalues_after_n_iterations((2 .* λ .- λ.^2); n=(n-1))
end

"""
    hyperpower_extreme_eigenvalues(P::H; n::Int=1, tol::T=√eps(T)) where {Dim,T,H<:HyperPowerPreconditioner{Dim,T}}

Compute the extreme eigenvalues of `inv(Pₙ)A`, i.e. return a tuple `(λmin, λmax)` in `n`th iteration.

In the case of eigenvalue clustering, i.e. around the value of 1, it might be necessary to choose lower `tol`.

Example:
```
julia
julia> λ₀ = hyperpower_extreme_eigenvalues(P; n=0, tol=10e-5)
(0.023145184193109333, 1.9749588237991926)

julia> λ₁ = hyperpower_extreme_eigenvalues(P; n=1, tol=10e-5)
(0.045754651575638426, 0.9999899575917253)

julia> hyperpower_extreme_eigenvalues(P; n=2, tol=10e-5)
(0.08941582990237358, 0.9999916150622586)

julia> hyperpower_extreme_eigenvalues(P; n=3, tol=10e-5)
(0.1708364549356042, 1.0000016638364837)

julia> hyperpower_extreme_eigenvalues(P; n=4, tol=10e-5)
(0.3124877477755778, 0.9999984021406245)

julia> hyperpower_extreme_eigenvalues(P; n=5, tol=10e-5)
(0.5273271247385951, 0.9999962548752878)

julia> hyperpower_eigenvalues_after_n_iterations(λ₁; n=2)
(0.17083641016503434, 1.0)

julia> hyperpower_eigenvalues_after_n_iterations(λ₁; n=3)
(0.31248774129199286, 1.0)

julia> hyperpower_eigenvalues_after_n_iterations(λ₁; n=4)
(0.5273268941262143, 1.0)

julia> hyperpower_eigenvalues_after_n_iterations(λ₀; n=1)
(0.0457546604483684, 0.049592390484542115)
```
"""
function hyperpower_extreme_eigenvalues(P::H; n::Int=1, tol::T=√eps(T)) where {Dim,T,H<:HyperPowerPreconditioner{Dim,T}}
    # get initial preconditioner
    X = hyperpower_initial_preconditioner(P)
    for k in Base.OneTo(n)
        X = HyperPowerPreconditioner(P.A, X)
    end

    # this may fail to converge if all eigenvalues are close to eachother!
    sr, srlog = partialschur(X * P.A; nev=1, tol=tol, which=:SR)
    lm, lmlog = partialschur(X * P.A; nev=1, tol=tol, which=:LM)
    @assert srlog.converged == lmlog.converged == true

    # get eigenvalues
    λmin = sr.eigenvalues[1]
    λmax = lm.eigenvalues[1]

    # sanity check of eigenvalue computation
    @assert abs(real(λmin)) - norm(λmin) < eps(T)
    @assert abs(real(λmax)) - norm(λmax) < eps(T)

    # get only real parts (imaginary component == 0)
    λmin = real(λmin)
    λmax = real(λmax)

    return λmin, λmax
end

"""
    LinearAlgebra.isposdef(P::H)

Returns `true` if `σ(inv(P₀)A) ⊂ (0,2)` and `false` otherwise.
"""
function LinearAlgebra.isposdef(P::H) where {Dim,T,H<:HyperPowerPreconditioner{Dim,T}}
    λmin, λmax = hyperpower_extreme_eigenvalues(P; n=0, tol=10e-5)
    λmin > 0.0 && λmax < 2.0
end

Base.size(P::HyperPowerPreconditioner) = size(P.A)
LinearAlgebra.issymmetric(::HyperPowerPreconditioner) = true
LinearAlgebra.ishermitian(P::HyperPowerPreconditioner) = isposdef(P)


struct BlockDiagonalPreconditioner{Dim,T} <: Preconditioner{Dim,T}
    L::LinearMaps.BlockDiagonalMap{T}
end
Base.size(P::BlockDiagonalPreconditioner) = size(P.L)
LinearAlgebra.issymmetric(::BlockDiagonalPreconditioner) = true
LinearAlgebra.isposdef(::BlockDiagonalPreconditioner) = true
LinearAlgebra.ishermitian(P::BlockDiagonalPreconditioner) = true
function LinearMaps._unsafe_mul!(b::AbstractVecOrMat, P::BlockDiagonalPreconditioner, v::AbstractVector)
    mul!(b, P.L, v)
end

struct ApproximateKroneckerInverse{Dim,T} <: Preconditioner{Dim,T}
    K::KroneckerProductAggregate{T}
    function ApproximateKroneckerInverse(K::KroneckerProductAggregate{T}; tol::T=10e-3, niter::Int=2, rank::Int=1) where {T}
        Dim = length(K.K[1].data)
        C, D = als_rank_q_approx_inv(K; tol=tol, niter=niter, q=rank)
        K⁻¹ = KroneckerProductAggregate(ntuple(k -> KroneckerProduct((C[k]+C[k]')/2, (D[k]+D[k]')/2; reverse=false) , rank)...)
        return new{Dim,T}(K⁻¹)
    end
end

Base.size(P::ApproximateKroneckerInverse) = size(P.K)
LinearAlgebra.issymmetric(::ApproximateKroneckerInverse) = true
LinearAlgebra.ishermitian(::ApproximateKroneckerInverse) = true
LinearAlgebra.isposdef(::ApproximateKroneckerInverse) = true

function LinearMaps._unsafe_mul!(b::AbstractVecOrMat, P::ApproximateKroneckerInverse, v::AbstractVector)
    mul!(b, P.K, v)
end