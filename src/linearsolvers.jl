export LinearSolver, TaigaCG, TaigaPCG, TaigaIPCG, linsolve!
export LinearSolverStatistics

# minimal reference implementations of basic linear solvers
# to be used with caution! 

"""
    abstract type LinearSolver{T} end

Concrete linear solvers derive from this abstract type.
"""
abstract type LinearSolver{T} end

function Base.show(io::IO, solver::S) where {S<:LinearSolver}
    atol = solver.atol
    rtol = solver.rtol
    itmax = solver.itmax
    tname = Base.typename(S).name
    print(io, "Linear solver of type $tname ")
    print(io, "(atol=$atol, rtol=$rtol, itmax=$itmax)")
end

"""
    linsolve!

Linear solvers implement this method. Typically,
the syntax is `linsolve!(solver, rhs; kwargs...)`
"""
function linsolve! end

"""
    LinearSolverStatistics{S <: LinearSolver}

A container for solver statistics. Contains a dictionary `data`.
Keys in `data` are used to mimic actual fields in this container,
see [`Base.propertynames`](@ref). Each property can be accessed
using [`Base.setproperty!`](@ref) and [`Base.getproperty`](@ref),
or `stats.propname`. The dictonary itself is not accessable
via `A.data`!

# Fields:
- `data::Dict{Symbol, Any}`: dictionary containing statistics
"""
struct LinearSolverStatistics{S<:LinearSolver}
    data::Dict{Symbol,Any}
end

"""
    LinearSolverStatistics(::Type{S}) where {S<:LinearSolver}

Default [`LinearSolverStatistics`](@ref) constructor
for one of [`LinearSolver`](@ref) types.

Following statistics are defined by default:

- `:converged` 
- `:status`
- `:residual`
- `:residual_norm_x₀`
- `:niter`
"""
function LinearSolverStatistics(::Type{S}) where {S<:LinearSolver}
    stats = LinearSolverStatistics{S}(Dict{Symbol,Any}())
    setproperty!(stats, :converged, false)
    setproperty!(stats, :status, "")
    setproperty!(stats, :residual, 0.0)
    setproperty!(stats, :niter, 0)
    return stats
end

"""
    Base.propertynames(A::LinearSolverStatistics)

Returns all property names of `A` (keys in `A.data`!). 
"""
Base.propertynames(A::LinearSolverStatistics) = keys(getfield(A, :data))

"""
    Base.setproperty!(A::LinearSolverStatistics, field::Symbol, value)

Sets `field` in `A` (actually `A.data[field]`) to `value`.

# Arguments:
- `A`: statistics container
- `field`: field to set (a key in `data`)
- `value`: value to set
"""
Base.setproperty!(A::LinearSolverStatistics, field::Symbol, value) = getfield(A, :data)[field] = value

"""
    Base.getproperty(A::LinearSolverStatistics, field::Symbol)

Returns `field` in `A` (actually the value associated with key `field` in `A.data`).
"""
Base.getproperty(A::LinearSolverStatistics, field::Symbol) = getfield(A, :data)[field]

function Base.show(io::IO, stats::LinearSolverStatistics{S}) where {S}
    print(io, "Linear solver statistics for ")
    printstyled(io, "$S", color=:green, bold=true)
    println(io, ":")
    pretty_table(
        io,
        getfield(stats, :data),
        column_labels=["Metric", "Value"],
        minimum_data_column_widths=[16, 80],
        fit_table_in_display_horizontally=true,
        fit_table_in_display_vertically=true,
        alignment=:l
    )
end

"""
    update_linear_solver_statistics!(solver::S)

For `η = √rᵀr` and `η₀ =  √r₀ᵀr₀` and check `η < atol`, `η₀ < rtol` and
if number of iterations is equal to `itmax`. Sets `solver.stats.converged`
and `solver.stats.status` accordingly.
"""
function update_linear_solver_statistics!(solver::S, ϵ) where {S<:LinearSolver}
    n, r, stats = size(solver.A, 2), solver.r, solver.stats
    γ = kdotr(n, r, r)
    stats.converged = sqrt(γ) ≤ ϵ

    stats.status = stats.converged ? "converged: " : "not converged: "

    if solver.stats.niter < solver.itmax
        stats.status *= "niter < itmax"
    elseif solver.stats.niter == solver.itmax
        stats.status *= "niter == itmax"
    end

    if stats.converged == false && solver.stats.niter != solver.itmax
        stats.status *= ", unexpected behaviour!"
    end
end

function reset_linear_solver_statistics!(solver::S) where {S<:LinearSolver}
    solver.stats.niter = 0
    solver.stats.status = ""
    solver.stats.converged = false
    solver.history && deleteat!(solver.stats.history, eachindex(solver.stats.history))
end

"""
    TaigaCG{T, L}

Linear solver implementing the Conjugate Gradient method.
"""
struct TaigaCG{T,L} <: LinearSolver{T}
    A::L
    x::Vector{T}
    p::Vector{T}
    Ap::Vector{T}
    r::Vector{T}
    atol::T
    rtol::T
    itmax::Int
    stats::LinearSolverStatistics{TaigaCG}
    history::Bool
    function TaigaCG(A::L; history=false, atol=√eps(eltype(A)), rtol=√eps(eltype(A)), itmax=200) where {L}
        n = size(A, 2)
        T = eltype(A)

        x = zeros(n)
        p = zeros(n)
        Ap = zeros(n)
        r = zeros(n)
        stats = LinearSolverStatistics(TaigaCG)
        history && setproperty!(stats, :history, Vector{T}())
        new{T,L}(A, x, p, Ap, r, atol, rtol, itmax, stats, history)
    end
end

"""
    linsolve!(solver::TaigaCG{T, L}, b::Vector{T}; x0::Vector{T} = zeros(size(solver.A, 2)))

Solve the linear system `A x = b` using [`TaigaCG`](@ref).

# Arguments:
- `solver`: solver context
- `b`: rhs vector
- `x0`: initial guess
"""
function linsolve!(solver::TaigaCG{T,L}, b::Vector{T}; x0::Vector{T}=zeros(size(solver.A, 2))) where {T,L}
    r, A, p, Ap, x = solver.r, solver.A, solver.p, solver.Ap, solver.x
    atol, rtol, itmax = solver.atol, solver.rtol, solver.itmax
    stats = solver.stats

    niter = 0
    n = size(A, 2)

    # initialize solution vector
    copyto!(x, x0)

    # compute residual
    mul!(r, A, x)
    kaxpby!(n, one(T), b, -one(T), r)

    # compute squared residual norm
    γ = kdotr(n, r, r)
    stats.residual = sqrt(γ)

    # convergence crit (compare to Krylov.jl)
    ϵ = atol + rtol * sqrt(γ) 

    # reset stats
    reset_linear_solver_statistics!(solver)
    solver.history && push!(stats.history, sqrt(γ))
    update_linear_solver_statistics!(solver, ϵ)
    
    # return if x0 good enough
    sqrt(γ) ≤ ϵ && return x, stats

    # set initial direction
    copyto!(p, r)

    # iterative solve
    while niter < itmax
        mul!(Ap, A, p)
        α = γ / kdotr(n, p, Ap)
        kaxpy!(n, α, p, x)
        β = 1 / γ
        kaxpy!(n, -α, Ap, r)
        γ = kdotr(n, r, r)

        niter += 1
        solver.history && push!(stats.history, sqrt(γ))
        sqrt(γ) ≤ ϵ && break

        β *= γ
        kaxpby!(n, one(T), r, β, p)
    end

    # update solver statistics
    stats.niter = niter
    stats.residual = sqrt(γ)
    update_linear_solver_statistics!(solver, ϵ)
    return x, stats
end

"""
    TaigaPCG{T, L, P}

Linear solver implementing the preconditioned Conjugate Gradient method.
"""
struct TaigaPCG{T,L,P} <: LinearSolver{T}
    A::L
    M::P
    x::Vector{T}
    p::Vector{T}
    Ap::Vector{T}
    r::Vector{T}
    z::Vector{T}
    atol::T
    rtol::T
    itmax::Int
    stats::LinearSolverStatistics{TaigaPCG}
    history::Bool
    function TaigaPCG(A::L, M::P; history=false, atol=√eps(eltype(A)), rtol=√eps(eltype(A)), itmax=200) where {L,P}
        @assert eltype(A) == eltype(M)
        @assert size(A) == size(M)
        n = size(A, 2)
        T = eltype(A)

        x = zeros(n)
        Δx = zeros(n)
        p = zeros(n)
        Δp = zeros(n)
        Ap = zeros(n)
        r = zeros(n)
        z = zeros(n)
        stats = LinearSolverStatistics(TaigaPCG)
        history && setproperty!(stats, :history, Vector{T}())
        new{T,L,P}(A, M, x, p, Ap, r, z, atol, rtol, itmax, stats, history)
    end
end

"""
    linsolve!(solver::TaigaPCG{T, L}, b::Vector{T}; x0::Vector{T} = zeros(size(solver.A, 2)))

Solve the linear system `A x = b` using [`TaigaPCG`](@ref).

# Arguments:
- `solver`: solver context
- `b`: rhs vector
- `x0`: initial guess
"""
function linsolve!(solver::TaigaPCG{T,L}, b::Vector{T}; x0::Vector{T}=zeros(size(solver.A, 2))) where {T,L}
    r, z, A, M, p, Ap, x, = solver.r, solver.z, solver.A, solver.M, solver.p, solver.Ap, solver.x
    atol, rtol, itmax = solver.atol, solver.rtol, solver.itmax
    stats = solver.stats

    niter = 0
    n = size(A, 2)

    # initialize solution vector
    copyto!(x, x0)

    # compute residual
    mul!(r, A, x)
    kaxpby!(n, one(T), b, -one(T), r)

    # compute squared residual norm
    γ = kdotr(n, r, r)
    stats.residual = sqrt(γ)

    # convergence crit (compare to Krylov.jl)
    ϵ = atol + rtol * sqrt(γ) 

    # reset stats
    reset_linear_solver_statistics!(solver)
    solver.history && push!(stats.history, sqrt(γ))
    update_linear_solver_statistics!(solver, ϵ)
    
    # return if x0 good enough
    sqrt(γ) ≤ ϵ && return x, stats

    # apply preconditioner to residual
    mul!(z, M, r)

    # set initial direction
    copyto!(p, z)

    # iterative solve
    while niter < itmax
        mul!(Ap, A, p)
        δ = kdotr(n, r, z)
        α = δ / kdotr(n, p, Ap)
        kaxpy!(n, α, p, x)
        β = 1 / δ
        kaxpy!(n, -α, Ap, r)
        γ = kdotr(n, r, r)

        niter += 1
        solver.history && push!(stats.history, sqrt(γ))
        sqrt(γ) ≤ ϵ && break

        mul!(z, M, r)
        δ = kdotr(n, r, z)
        β *= δ
        kaxpby!(n, one(T), z, β, p)
    end

    # update solver statistics
    stats.niter = niter
    stats.residual = sqrt(γ)
    update_linear_solver_statistics!(solver, ϵ)
    return x, stats   
end

"""
    TaigaIPCG{T, L, P}

Linear solver implementing the inexactly preconditioned Conjugate Gradient method.

# References
1. Gene H. Golub and Qiang Ye. Inexact preconditioned conjugate gradient method with inner-outer iteration. SIAM J. Sci. Comput., 21(4):1305–1320, December 1999.
2. Andrew V Knyazev and Ilya Lashuk. Steepest descent and conjugate gradient methods with variable preconditioning. SIAM Journal on Matrix Analysis and Applications, 29(4):1267–1280, 2008.
"""
struct TaigaIPCG{T,L,P} <: LinearSolver{T}
    A::L
    M::P
    x::Vector{T}
    p::Vector{T}
    Ap::Vector{T}
    r::Vector{T}
    Δr::Vector{T}
    z::Vector{T}
    atol::T
    rtol::T
    itmax::Int
    stats::LinearSolverStatistics{TaigaIPCG}
    history::Bool
    function TaigaIPCG(A::L, M::P; history=false, atol=√eps(eltype(A)), rtol=√eps(eltype(A)), itmax=200) where {L,P}
        @assert eltype(A) == eltype(M)
        @assert size(A) == size(M)
        n = size(A, 2)
        T = eltype(A)

        x = zeros(n)
        p = zeros(n)
        Ap = zeros(n)
        r = zeros(n)
        Δr = zeros(n)
        z = zeros(n)
        stats = LinearSolverStatistics(TaigaIPCG)
        history && setproperty!(stats, :history, Vector{T}())
        new{eltype(A),L,P}(A, M, x, p, Ap, r, Δr, z, atol, rtol, itmax, stats, history)
    end
end

"""
    linsolve!(solver::TaigaCG{T, L}, b::Vector{T}; x0::Vector{T} = zeros(size(solver.A, 2)))

Solve the linear system `A x = b` using [`TaigaIPCG`](@ref).

# Arguments:
- `solver`: solver context
- `b`: rhs vector
- `x0`: initial guess
"""
function linsolve!(solver::TaigaIPCG{T,L}, b::Vector{T}; x0::Vector{T}=zeros(size(solver.A, 2))) where {T,L}
    r, Δr, z, A, M, p, Ap, x = solver.r, solver.Δr, solver.z, solver.A, solver.M, solver.p, solver.Ap, solver.x
    atol, rtol, itmax = solver.atol, solver.rtol, solver.itmax
    stats = solver.stats

    niter = 0
    n = size(A, 2)

    # initialize solution vector
    copyto!(x, x0)

    # compute residual
    mul!(r, A, x)
    kaxpby!(n, one(T), b, -one(T), r)

    # apply preconditioner to residual
    mul!(z, M, r)

    # compute squared residual norm
    γ = kdotr(n, r, r)
    δ = kdotr(n, z, r)
    stats.residual = sqrt(γ)

    # convergence crit (compare to Krylov.jl)
    ϵ = atol + rtol * sqrt(γ) 

    # reset stats
    reset_linear_solver_statistics!(solver)
    solver.history && push!(stats.history, sqrt(γ))
    update_linear_solver_statistics!(solver, ϵ)
    
    # return if x0 good enough
    sqrt(γ) ≤ ϵ && return x, stats

    # set initial direction
    copyto!(p, z)

    #iterative solve
    while niter < itmax
        mul!(Ap, A, p)
        α = kdotr(n, z, r) / kdotr(n, p, Ap)
        kaxpy!(n, α, p, x)

        copyto!(Δr, r) # Δr = r_{k-1}
        β = 1 / kdotr(n, z, r) # β = (z_{k-1}' r_{k-1})⁻¹
        kaxpy!(n, -α, Ap, r) # r_{k}
        γ = kdotr(n, r, r)

        niter += 1
        solver.history && push!(stats.history, sqrt(γ))
        sqrt(γ) ≤ ϵ && break

        mul!(z, M, r)
        kaxpby!(n, one(T), r, -one(T), Δr) # Δr = r_{k} - r_{k-1}
        δ = kdotr(n, z, Δr)
        β *= δ # β = (z_{k}' r_{k}) / (z_{k-1}' r_{k-1})
        kaxpby!(n, one(T), z, β, p)
    end

    # update solver statistics
    stats.niter = niter
    stats.residual = sqrt(γ)
    update_linear_solver_statistics!(solver, ϵ)
    return x, stats
end