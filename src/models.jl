export Model
export LinearOperator, LinearOperatorApproximation
export linear_operator, linear_operator_approximation
export forcing!, forcing
export apply_particular_solution
export Constrained

"""
    abstract type Model

Concrete models derive from this abstract type.
"""
abstract type Model{Dim,T} end

"""
    abstract type LinearOperator{Dim,T}

Concrete linear operators derive from this abstract type.
"""
abstract type LinearOperator{Dim,T} <: LinearMap{T} end

"""
    abstract type LinearOperatorApproximation{Dim,T}

Concrete linear operator approximations derive from this abstract type.
"""
abstract type LinearOperatorApproximation{Dim,T} <: LinearMap{T} end

"""
    linear_operator()

Returns a linear operator for a model. Each model module implements this.
"""
function linear_operator end

"""
    linear_operator_approximation()

Returns a linear operator approximation for a model. Some model modules implements this.
Linear operator approximations are cheap to apply and useful in the context of preconditioning.
"""
function linear_operator_approximation end

"""
    forcing!

Each model module implements its own method for [`forcing!`](@ref), which updates the
right hand side vector.
"""
function forcing! end

"""
    forcing

Each model module implements its own method for [`forcing`](@ref), which returns the
right hand side vector.
"""
function forcing end

"""
    apply_particular_solution

Each model module implements its own method for [`apply_particular_solution`](@ref), which
adds particular solution to homogeneous solution.
"""
function apply_particular_solution end

function Base.show(io::IO, A::L) where {L<:LinearOperator}
    sz = size(A)
    print(io, "Linear operator of type $L with size $sz")
end

function Base.show(io::IO, A::L) where {L<:LinearOperatorApproximation}
    sz = size(A)
    print(io, "Linear operator approximation of type $L with size $sz")
end

"""
    Constrained{x}

Can be used as a constrained/unconstrained flag. Similar to `Val` but takes only
booleans as `x`.
"""
struct Constrained{x} end
Constrained(x::T) where {T<:Bool} = Constrained{x}()