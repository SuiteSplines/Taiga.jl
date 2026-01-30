# Example module showcasing basic interface of a linear model
module ExampleModule
    using Taiga
    export ExampleModel

    struct ExampleModel{Dim,T} <: Model{Dim,T} end
    
    struct LinearOperator{Dim,T} <: Taiga.LinearOperator{Dim,T} end

    struct LinearOperatorApproximation{Dim,T} <: Taiga.LinearOperatorApproximation{Dim,T} end

    Taiga.linear_operator(model::ExampleModel) = LinearOperator{3,Float64}()
    Taiga.linear_operator_approximation(model::ExampleModel) = LinearOperatorApproximation{3,Float64}()
    Base.size(::LinearOperator) = (42,42)
    Base.size(::LinearOperatorApproximation) = (42,42)

    function Taiga.LinearMaps._unsafe_mul!(b::AbstractVecOrMat, L::LinearOperator, v::AbstractVector)
        b .= v
    end

    function Taiga.LinearMaps._unsafe_mul!(b::AbstractVecOrMat, L::LinearOperatorApproximation, v::AbstractVector)
        b .= (0.5 .* v)
    end

    function Taiga.forcing!(b::AbstractVector, model::ExampleModel{Dim}) where {Dim}
        b .= π
    end

    function Taiga.forcing(model::ExampleModel{Dim}) where {Dim}
        2π .* ones(42)
    end
end
# Taiga integration
using .ExampleModule
export ExampleModule, ExampleModel
