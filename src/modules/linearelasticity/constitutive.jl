"""
    BMatrix{Dim, F}

Struct holding B-matrices (strain-displacement matrices). Supports linear indexing.
"""
struct BMatrix{Dim,F}
    data::NTuple{Dim,F}
end

function BMatrix{2}()
    E1 = @SMatrix  [1 0; 
                    0 0;
                    0 1]
    E2 = @SMatrix  [0 0; 
                    0 1;
                    1 0]
    F = typeof(E1)
    return BMatrix{2,F}((E1, E2))
end

function BMatrix{3}()
    E1 = @SMatrix  [1 0 0; 
        0 0 0;
        0 0 0;
        0 0 0;
        0 0 1;
        0 1 0]

    E2 = @SMatrix  [0 0 0; 
        0 1 0;
        0 0 0;
        0 0 1;
        0 0 0;
        1 0 0]

    E3 = @SMatrix  [0 0 0; 
        0 0 0;
        0 0 1;
        0 1 0;
        1 0 0;
        0 0 0]
    
    F = typeof(E1)
    return BMatrix{3,F}((E1, E2, E3))
end
function Base.getindex(B::BMatrix{Dim}, i::Int) where {Dim}
    @assert i <= Dim
    return B.data[i]
end

# getindex and setindex! method for 3d material matrix
@eval function Base.getindex(Y::EvaluationSet{6,6}, I...)
    return @SMatrix [Y.data[i,j][I...] for i in 1:6, j in 1:6]
end
@eval function Base.setindex!(Y::EvaluationSet{6,6}, v, I...)
    for j in 1:6
        for i in 1:6
            Y.data[i,j][I...] = v[i,j]
        end
    end
end


"""
    abstract type Material{Dim,S} <: AbstractMapping{Dim,S,S} end

All linear elastic materials derive from this. Materials are mappings
in `Dim × S × S`.
"""
abstract type Material{Dim,S} <: AbstractMapping{Dim,S,S} end

"""
    PlaneStress <: Material{2,3}

Plane stress material.

# Fields:
- `E::Function`: Young's modulus
- `ν::Function`: Poisson's ratio
"""
struct PlaneStress <: Material{2,3}
    E::Function
    ν::Function
    function PlaneStress(E::Function, ν::Function)
        args = ι(0, dim=2)
        @assert applicable(E, args...)
        @assert applicable(ν, args...)
        new(E,ν)
    end
end

"""
    PlaneStrain <: Material{2,3}

Plane strain material.

# Fields:
- `E::Function`: Young's modulus
- `ν::Function`: Poisson's ratio
"""
struct PlaneStrain <: Material{2,3}
    E::Function
    ν::Function
    function PlaneStrain(E::Function, ν::Function)
        args = ι(0, dim=2)
        @assert applicable(E, args...)
        @assert applicable(ν, args...)
        new(E,ν)
    end
end

"""
    Isotropic <: Material{3,6}

Isotropic material.

# Fields:
- `E::Function`: Young's modulus
- `ν::Function`: Poisson's ratio
"""
struct Isotropic <: Material{3,6}
    E::Function
    ν::Function
    function Isotropic(E::Function, ν::Function)
        args = ι(0, dim=3)
        @assert applicable(E, args...)
        @assert applicable(ν, args...)
        new(E,ν)
    end
end

"""
    Orthotropic <: Material{3,6}

Orthotropic material.

# Fields:
- `E₁::Function`: Young's modulus
- `E₂::Function`: Young's modulus
- `E₃::Function`: Young's modulus
- `ν₂₃::Function`: Poisson's ratio
- `ν₃₂::Function`: Poisson's ratio
- `ν₁₃::Function`: Poisson's ratio
- `ν₃₁::Function`: Poisson's ratio
- `ν₁₂::Function`: Poisson's ratio
- `ν₂₁::Function`: Poisson's ratio
- `G₂₃::Function`: Shear modulus
- `G₁₃::Function`: Shear modulus
- `G₁₂::Function`: Shear modulus
"""
struct Orthotropic <: Material{3,6}
    E₁::Function
    E₂::Function
    E₃::Function
    ν₂₃::Function
    ν₃₂::Function
    ν₁₃::Function
    ν₃₁::Function
    ν₁₂::Function
    ν₂₁::Function
    G₂₃::Function
    G₁₃::Function
    G₁₂::Function
    function Orthotropic(E₁::Function, E₂::Function, E₃::Function, ν₂₃::Function, ν₃₂::Function, ν₁₃::Function, ν₃₁::Function, ν₁₂::Function, ν₂₁::Function, G₂₃::Function, G₁₃::Function, G₁₂::Function)
        args = ι(0, dim=3)
        @assert applicable(E₁, args...)
        @assert applicable(E₂, args...)
        @assert applicable(E₃, args...)
        @assert applicable(ν₂₃, args...)
        @assert applicable(ν₃₂, args...)
        @assert applicable(ν₁₃, args...)
        @assert applicable(ν₃₁, args...)
        @assert applicable(ν₁₂, args...)
        @assert applicable(ν₂₁, args...)
        @assert applicable(G₂₃, args...)
        @assert applicable(G₁₃, args...)
        @assert applicable(G₁₂, args...)
        new(E₁, E₂, E₃, ν₂₃, ν₃₂, ν₁₃, ν₃₁, ν₁₂, ν₂₁, G₂₃, G₁₃, G₁₂)
    end
end

"""
    material_matrix(::Type{PlaneStress}, E, ν)

Returns the elasticity tensor in Voigt notation.
"""
function material_matrix(::Type{PlaneStress}, E, ν)
    μ = E / (2 * (1 + ν))
    λ = ν * E / ((1 + ν) * (1 - 2ν))
    λ̄ = 2*λ*μ / (λ + 2μ)
    [λ̄+2μ    λ̄       0.0
     λ̄       λ̄+2μ    0.0 
     0.0     0.0     μ  ];
end

"""
    material_matrix(::Type{PlaneStrain}, E, ν)

Returns the elasticity tensor in Voigt notation.
"""
function material_matrix(::Type{PlaneStrain}, E, ν)
    μ = E / (2 * (1 + ν))
    λ = ν * E / ((1 + ν) * (1 - 2ν))
    [λ+2μ    λ       0.0
     λ       λ+2μ    0.0 
     0.0     0.0     μ  ];
end

"""
    material_matrix(::Type{Isotropic}, E, ν)

Returns the elasticity tensor in Voigt notation.
"""
function material_matrix(::Type{Isotropic}, E, ν)
    μ = E / (2 * (1 + ν))
    λ = ν * E / ((1 + ν) * (1 - 2ν))
    [λ+2μ    λ       λ       0.0     0.0     0.0;
     λ       λ+2μ    λ       0.0     0.0     0.0; 
     λ       λ       λ+2μ    0.0     0.0     0.0;
     0.0     0.0     0.0     μ       0.0     0.0;
     0.0     0.0     0.0     0.0     μ       0.0;
     0.0     0.0     0.0     0.0     0.0     μ];
end

"""
    material_matrix(::Type{Orthotropic}, E₁, E₂, E₃, ν₂₃, ν₃₂, ν₁₃, ν₃₁, ν₁₂, ν₂₁, G₂₃, G₁₃, G₁₂)

Returns the elasticity tensor in Voigt notation.
"""
function material_matrix(::Type{Orthotropic}, E₁, E₂, E₃, ν₂₃, ν₃₂, ν₁₃, ν₃₁, ν₁₂, ν₂₁, G₂₃, G₁₃, G₁₂)
    D = 1 - ν₁₂ * ν₂₁ - ν₁₃ * ν₃₁ - ν₂₃ * ν₃₂ - 2 * ν₁₂ * ν₂₃ * ν₃₁
    [(1 - ν₂₃ * ν₃₂) / D * E₁     (ν₂₁ + ν₂₃ * ν₃₁) / D * E₁     (ν₃₁ + ν₃₂ * ν₂₁) / D * E₁     0    0    0;
     (ν₁₂ + ν₁₃ * ν₃₂) / D * E₂   (1 - ν₁₃ * ν₃₁) / D * E₂       (ν₃₂ + ν₃₁ * ν₁₂) / D * E₂     0    0    0;
     (ν₁₃ + ν₁₂ * ν₂₃) / D * E₃   (ν₂₃ + ν₂₁ * ν₁₃) / D * E₃     (1 - ν₁₂ * ν₂₁) / D * E₃       0    0    0;
     0                            0                              0                              G₂₃  0    0;
     0                            0                              0                              0    G₁₃  0;
     0                            0                              0                              0    0    G₁₂]
end

for T in (:PlaneStrain, :PlaneStress)
    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{3,3}, x::CartesianProduct{2}, material::T) where {T<:$T}
        for k in 1:length(x)  
            E = material.E(x[k]...)
            ν = material.ν(x[k]...)
            y[k] = material_matrix(T, E, ν);
        end
        return y
    end
    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{3,3}, x::EvaluationSet{1,2}, material::T) where {T<:$T}
        for k in 1:length(x)  
            E = material.E(x[k]...)
            ν = material.ν(x[k]...)
            y[k] = material_matrix(T, E, ν);
        end
        return y
    end
end
@eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{6,6}, x::CartesianProduct{3}, material::T) where {T<:Isotropic}
    for k in 1:length(x)  
        E = material.E(x[k]...)
        ν = material.ν(x[k]...)
        y[k] = material_matrix(T, E, ν);
    end
    return y
end
@eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{6,6}, x::EvaluationSet{1,3}, material::T) where {T<:Isotropic}
    for k in 1:length(x)  
        E = material.E(x[k]...)
        ν = material.ν(x[k]...)
        y[k] = material_matrix(T, E, ν);
    end
    return y
end
@eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{6,6}, x::CartesianProduct{3}, material::T) where {T<:Orthotropic}
    for k in 1:length(x)  
        E₁ = material.E₁(x[k]...)
        E₂ = material.E₂(x[k]...)
        E₃ = material.E₃(x[k]...)
        ν₂₃ = material.ν₂₃(x[k]...)
        ν₃₂ = material.ν₃₂(x[k]...)
        ν₁₃ = material.ν₁₃(x[k]...)
        ν₃₁ = material.ν₃₁(x[k]...)
        ν₁₂ = material.ν₁₂(x[k]...)
        ν₂₁ = material.ν₂₁(x[k]...)
        G₂₃ = material.G₂₃(x[k]...)
        G₁₃ = material.G₁₃(x[k]...)
        G₁₂ = material.G₁₂(x[k]...)
        y[k] = material_matrix(T, E₁, E₂, E₃, ν₂₃, ν₃₂, ν₁₃, ν₃₁, ν₁₂, ν₂₁, G₂₃, G₁₃, G₁₂);
    end
    return y
end
@eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{6,6}, x::EvaluationSet{1,3}, material::T) where {T<:Orthotropic}
    for k in 1:length(x)  
        E₁ = material.E₁(x[k]...)
        E₂ = material.E₂(x[k]...)
        E₃ = material.E₃(x[k]...)
        ν₂₃ = material.ν₂₃(x[k]...)
        ν₃₂ = material.ν₃₂(x[k]...)
        ν₁₃ = material.ν₁₃(x[k]...)
        ν₃₁ = material.ν₃₁(x[k]...)
        ν₁₂ = material.ν₁₂(x[k]...)
        ν₂₁ = material.ν₂₁(x[k]...)
        G₂₃ = material.G₂₃(x[k]...)
        G₁₃ = material.G₁₃(x[k]...)
        G₁₂ = material.G₁₂(x[k]...)
        y[k] = material_matrix(T, E₁, E₂, E₃, ν₂₃, ν₃₂, ν₁₃, ν₃₁, ν₁₂, ν₂₁, G₂₃, G₁₃, G₁₂);
    end
    return y
end
