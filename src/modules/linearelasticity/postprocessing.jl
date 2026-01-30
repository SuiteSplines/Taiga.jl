"""
    ϵ_voigt(A::SMatrix) -> SVector
    ϵ_voigt(v::SVector) -> SMatrix

Convert between standard and Voigt notation for strain.

- `ϵ_voigt(A::SMatrix{2,2})`: Converts a 2×2 strain tensor to a 3-component Voigt vector.
- `ϵ_voigt(A::SMatrix{3,3})`: Converts a 3×3 strain tensor to a 6-component Voigt vector.
- `ϵ_voigt(v::SVector{3})`: Converts a 3-component Voigt strain vector to a 2×2 strain tensor.
- `ϵ_voigt(v::SVector{6})`: Converts a 6-component Voigt strain vector to a 3×3 strain tensor.

Shear components are multiplied by `2` when converting to Voigt notation.
"""
ϵ_voigt

"""
    σ_voigt(A::SMatrix) -> SVector
    σ_voigt(v::SVector) -> SMatrix

Convert between standard and Voigt notation for stress.

- `σ_voigt(A::SMatrix{2,2})`: Converts a 2×2 stress tensor to a 3-component Voigt vector.
- `σ_voigt(A::SMatrix{3,3})`: Converts a 3×3 stress tensor to a 6-component Voigt vector.
- `σ_voigt(v::SVector{3})`: Converts a 3-component Voigt stress vector to a 2×2 stress tensor.
- `σ_voigt(v::SVector{6})`: Converts a 6-component Voigt stress vector to a 3×3 stress tensor.

Shear components remain unchanged in Voigt notation.
"""
σ_voigt

@inline ϵ_voigt(A::SMatrix{2,2,T}) where {T} =  @SVector [  A[1,1]
                                                            A[2,2]
                                                          2*A[1,2]]

@inline ϵ_voigt(A::SMatrix{3,3,T}) where {T} = @SVector [  A[1,1]
                                                           A[2,2]
                                                           A[3,3]
                                                         2*A[2,3]
                                                         2*A[1,3]
                                                         2*A[1,2]]

@inline ϵ_voigt(v::SVector{3,T}) where {T} = @SMatrix [    v[1] 0.5*v[3]
                                                       0.5*v[3]     v[2]]

@inline ϵ_voigt(v::SVector{6,T}) where {T} = @SMatrix [    v[1] 0.5*v[6] 0.5*v[5]
                                                       0.5*v[6]     v[2] 0.5*v[4]
                                                       0.5*v[5] 0.5*v[4]     v[3]]

@inline σ_voigt(A::SMatrix{2,2,T}) where {T} = @SVector [A[1,1]
                                                         A[2,2]
                                                         A[1,2]]

@inline σ_voigt(A::SMatrix{3,3,T}) where {T} = @SVector [A[1,1]
                                                         A[2,2]
                                                         A[3,3]
                                                         A[2,3]
                                                         A[1,3]
                                                         A[1,2]]

@inline σ_voigt(v::SVector{3,T}) where {T} = @SMatrix [v[1] v[3]
                                                       v[3] v[2]]

@inline σ_voigt(v::SVector{6,T}) where {T} = @SMatrix [v[1] v[6] v[5]
                                                       v[6] v[2] v[4]
                                                       v[5] v[4] v[3]]

"""
    Strain{Dim, T1 <: GeometricMapping{Dim}, T2 <: Field{Dim, Dim}} <: AbstractMapping{Dim,Dim,Dim}

Mapping for evaluation of strain:
    
``\\boldsymbol \\varepsilon = \\frac 1 2 (\\nabla \\mathbf u + \\nabla \\mathbf u^T)``.

# Fields:
- `mapping::T1`: geometric mapping
- `displacement::T2`: displacement field
"""
struct Strain{Dim,T1<:GeometricMapping{Dim}, T2<:Field{Dim,Dim}} <: AbstractMapping{Dim,Dim,Dim}
    mapping::T1
    displacement::T2
    function Strain(mapping::GeometricMapping{Dim}, displacement::Field{Dim,Dim}) where Dim
        T1 = typeof(mapping)
        T2 = typeof(displacement)
        new{Dim,T1,T2}(mapping, displacement)
    end
end

IgaBase.standard_quadrature_rule(f, g::Strain) = IgaBase.standard_quadrature_rule(f, g.displacement)

for Dim in 2:3
    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{$Dim,$Dim}, x::CartesianProduct{$Dim}, ε::Strain{$Dim})
        mapping = ε.mapping
        displacement = ε.displacement

        @evaluate ∇F = Gradient(mapping)(x)
        @evaluate! y = Gradient(displacement)(x)
        
        for k in 1:length(x)  
            J = ∇F[k]
            ∇u = inv(J) * y[k] 
            y[k] = 0.5 * (∇u + ∇u')
        end
        return y
    end
end

"""
    CauchyStress{Dim, T <: Strain{Dim}} <: AbstractMapping{Dim,Dim,Dim}

Mapping for evaluation of Cauchy stress:
    
``\\boldsymbol \\sigma = \\mathbf C : \\boldsymbol \\varepsilon``.

# Fields:
- `strain::T`: [`Strain`](@ref)
- `material::Material{Dim}`: [`Material`](@ref)
"""
struct CauchyStress{Dim,T<:Strain{Dim}} <: AbstractMapping{Dim,Dim,Dim}
    strain::T
    material::Material{Dim}
    function CauchyStress(strain::T, material::Material{Dim}) where {Dim, T<:Strain{Dim}}
        new{Dim,T}(strain, material)
    end
end

IgaBase.standard_quadrature_rule(f, g::CauchyStress) = IgaBase.standard_quadrature_rule(f, g.strain.displacement)

for Dim in 2:3
    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, σ::EvaluationSet{$Dim,$Dim}, x::CartesianProduct{$Dim}, stress::CauchyStress{$Dim})
        strain = stress.strain
        material = stress.material  
        ϵ = σ

        @evaluate D = material(x)
        @evaluate! ϵ = strain(x)
        Id = SMatrix{$Dim,$Dim}(I)

        for k in 1:length(x)  
            #σ[k] = λ * tr(ε[k]) * Id + 2.0 * μ * ε[k]
            σ[k] = σ_voigt(D[k] * ϵ_voigt(ϵ[k]))
        end
        return σ
    end
end

"""
    VonMisesStress{Dim, T <: CauchyStress{Dim}}

Mapping for evaluation of von Mises stress.

```math
\\begin{aligned}
\\mathbf{s} &= \\boldsymbol{\\sigma} - \\frac{1}{2} \\mathrm{tr}(\\boldsymbol{\\sigma}) \\mathbf{I} \\\\
J_2 &= \\mathbf{s} : \\mathbf{s}) \\\\
\\boldsymbol{\\sigma}_{\\mathrm{vM}} &= \\sqrt{3 J_2}
\\end{aligned}
```

# Fields:
- `stress::T`: [`CauchyStress`](@ref)
"""
struct VonMisesStress{Dim,T<:CauchyStress{Dim}} <: AbstractMapping{Dim,1,1}
    stress::T
    function VonMisesStress(stress::T) where {Dim, T<:CauchyStress{Dim}}
        new{Dim,T}(stress)
    end
end

IgaBase.standard_quadrature_rule(f, g::VonMisesStress) = IgaBase.standard_quadrature_rule(f, g.stress.strain.displacement)

for Dim in 2:3
    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{1,1}, x::CartesianProduct{$Dim}, σᵥ::VonMisesStress{$Dim})
        stress = σᵥ.stress

        @evaluate σ = stress(x)
        Id = SMatrix{$Dim,$Dim}(I)

        for k in 1:length(x)  
            s = σ[k] - 1/3 * tr(σ[k]) * Id
            J₂ = 0.5 * dot(s,s)
            y[k] = sqrt(3J₂)
        end
        return y
    end
end
