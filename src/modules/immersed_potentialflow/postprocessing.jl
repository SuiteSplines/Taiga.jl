
"""
    abstract type PrimaryField end

Primary field indicator.
"""
abstract type PrimaryField end


"""
    StreamFunction <: PrimaryField

Stream function primary field.
"""
struct StreamFunction <: PrimaryField end

"""
    VelocityPotential <: PrimaryField

Velocity potential primary field.
"""
struct VelocityPotential <: PrimaryField end

"""
    Velocity{Dim, F <: PrimaryField, T1 <: GeometricMapping{Dim}, T2 <: Field{Dim}}

Velocity mapping. Supports potential as velocity potential or stream function.

# Fields:
- `mapping::T1`: geometric mapping
- `potential::T2`: potential
"""
struct Velocity{Dim,F<:PrimaryField,T1<:GeometricMapping{Dim},T2<:Field{Dim}} <: AbstractMapping{Dim,Dim,1}
    mapping::T1
    potential::T2
    function Velocity(mapping::GeometricMapping{Dim}, potential::Field{Dim}) where Dim
        T1 = typeof(mapping)
        T2 = typeof(potential)
        new{Dim,VelocityPotential,T1,T2}(mapping, potential)
    end
    function Velocity{StreamFunction}(mapping::GeometricMapping{Dim}, potential::Field{Dim}) where Dim
        T1 = typeof(mapping)
        T2 = typeof(potential)
        new{Dim,StreamFunction,T1,T2}(mapping, potential)
    end
end

for Dim in 2:3
    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{$Dim,1}, x::CartesianProduct{$Dim}, v::Velocity{$Dim,VelocityPotential})
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
function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{2,1}, x::CartesianProduct{2}, v::Velocity{2,StreamFunction})
    mapping = v.mapping
    potential = v.potential

    @evaluate ∇F = Gradient(mapping)(x)
    @evaluate! y = Gradient(potential)(x)
    T = @SMatrix [0 1; -1 0];
    
    for k in eachindex(x)  
        J = ∇F[k]
        ∇u = inv(J) * y[k] 
        y[k] = T * ∇u
    end
    return y
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

# what follows is not fast... it just works
function (p::Pressure)(x...)
    x = CartesianProduct(x...)
    @evaluate y = p(x)
    y.data[1][1]
end

function (v::Velocity{2})(x...)
    x = CartesianProduct(x...)
    @evaluate y = v(x)
    @SVector [y.data[1][1]; y.data[2][1]]
end

function (v::Velocity{3})(x...)
    x = CartesianProduct(x...)
    @evaluate y = v(x)
    @SVector [y.data[1][1]; y.data[2][1]; y.data[3][1]]
end

function IgaBase.project!(distancefun::AlgoimCallLevelSetFunction, field::Field{Dim,1}; onto, method::Type{GalerkinProjection}) where Dim
    IgaBase.project!(distancefun, field[1]; onto=onto, method=GalerkinProjection)
end

function IgaBase.project!(distancefun::AlgoimCallLevelSetFunction, field::Field{Dim,2}; onto, method::Type{GalerkinProjection}) where Dim
    f = ntuple(d -> ScalarFunction((x,y) -> onto(x,y)[d]), Dim)
    IgaBase.project!(distancefun, field[1]; onto=f[1], method=GalerkinProjection)
    IgaBase.project!(distancefun, field[2]; onto=f[2], method=GalerkinProjection)
end

function IgaBase.project!(distancefun::AlgoimCallLevelSetFunction, field::Field{Dim,3}; onto, method::Type{GalerkinProjection}) where Dim
    f = ntuple(d -> ScalarFunction((x,y,z) -> onto(x,y,z)[d]), Dim)
    IgaBase.project!(distancefun, field[1]; onto=f[1], method=GalerkinProjection)
    IgaBase.project!(distancefun, field[2]; onto=f[2], method=GalerkinProjection)
    IgaBase.project!(distancefun, field[3]; onto=f[3], method=GalerkinProjection)
end
