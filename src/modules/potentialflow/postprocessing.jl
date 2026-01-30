"""
    Velocity{Dim, T1 <: GeometricMapping{Dim}, T2 <: Field{Dim}}

Velocity mapping.

# Fields:
- `mapping::T1`: geometric mapping
- `potential::T2`: potential field
"""
struct Velocity{Dim,T1<:GeometricMapping{Dim}, T2<:Field{Dim}} <: AbstractMapping{Dim,Dim,1}
    mapping::T1
    potential::T2
    function Velocity(mapping::GeometricMapping{Dim}, potential::Field{Dim}) where Dim
        T1 = typeof(mapping)
        T2 = typeof(potential)
        new{Dim,T1,T2}(mapping, potential)
    end
end

for Dim in 2:3
    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{$Dim,1}, x::CartesianProduct{$Dim}, v::Velocity{$Dim})
        mapping = v.mapping
        potential = v.potential

        @evaluate ∇F = Gradient(mapping)(x)
        @evaluate! y = Gradient(potential)(x)
        
        for k in eachindex(x)  
            J = ∇F[k]
            ∇u = inv(J) * y[k] 
            y[k] = ∇u
        end
        return y
    end
end

"""
    Pressure{Dim, T1, T2 <: Velocity{Dim}}

Pressure mapping for uniform far field flow.

# Fields:
- `ρ::T1`: density
- `p::T1`: far field pressure
- `U::T1`: far field velocity magnitude
- `velocity::T2`: velocity field
"""
struct Pressure{Dim,T1,T2<:Velocity{Dim}} <: AbstractMapping{Dim,1,1}
    ρ::T1
    p::T1
    U::T1
    velocity::T2
    function Pressure(velocity::Velocity{Dim}, U::T1; ρ::T1=1.0, p::T1=1.0) where {Dim,T1}
        T2 = typeof(velocity)
        new{Dim,T1,T2}(ρ, p, U, velocity)
    end
end

for Dim in 2:3
    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{1,1}, x::CartesianProduct{$Dim}, pressure::Pressure{$Dim})
        ρ, p, U = pressure.ρ, pressure.p, pressure.U
        @evaluate V = pressure.velocity(x)
        for k in eachindex(x)  
            y[k] = 0.5 * ρ * (U^2 - sum(V[k].^2)) + p
        end
        return y
    end
end

IgaBase.standard_quadrature_rule(f, g::Velocity) = IgaBase.standard_quadrature_rule(f, g.potential)
IgaBase.standard_quadrature_rule(f, g::Pressure) = IgaBase.standard_quadrature_rule(f, g.velocity.potential)


#struct PointwiseError{Dim,T1<:GeometricMapping{Dim}, T2<:Field{Dim}, T3<:ScalarFunction{Dim}} <: AbstractMapping{Dim,1,1}
#    mapping::T1
#    potential::T2
#    reference::T3
#    function PointwiseError(mapping::GeometricMapping{Dim}, potential::Field{Dim}, reference::ScalarFunction{Dim}) where Dim
#        T1 = typeof(mapping)
#        T2 = typeof(potential)
#        T3 = typeof(reference)
#        new{Dim,T1,T2,T3}(mapping, potential, reference)
#    end
#end
#
#for Dim in 2:3
#    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{1,1}, x::CartesianProduct{$Dim}, e::PointwiseError{$Dim})
#        mapping = e.mapping
#        potential = e.potential
#        reference = e.reference
#
#        @evaluate X = mapping(x)
#        @evaluate uʰ = potential(x)
#        
#        for k in eachindex(x)  
#            y[k] = uʰ[k] - reference(X[k]...)
#        end
#        return y
#    end
#end