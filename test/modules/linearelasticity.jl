using Taiga
using StaticArrays
using LinearAlgebra
using SortedSequences
using CartesianProducts
using AbstractMappings
using UnivariateSplines
using NURBS
using SparseArrays

@testset "Voigt notation" begin
    # stress tensor in two dimensions
    A = @SMatrix [ 1.0 2.0
                   2.0 3.0 ]
    a = @SVector [ 1.0; 3.0; 4.0 ]
    @test LinearElasticity.ϵ_voigt(A) == a
    @test LinearElasticity.ϵ_voigt(a) == A

    # strain tensor in three dimensions
    A = @SMatrix [ 1.0 2.0 3.0
                   2.0 4.0 5.0
                   3.0 5.0 6.0 ]
    a = @SVector [ 1.0; 4.0; 6.0; 10.0; 6.0; 4.0 ]
    @test LinearElasticity.ϵ_voigt(A) == a
    @test LinearElasticity.ϵ_voigt(a) == A

    # stress tensor in two dimensions
    A = @SMatrix [ 1.0 2.0
                   2.0 3.0 ]
    a = @SVector [ 1.0; 3.0; 2.0 ]
    @test LinearElasticity.σ_voigt(A) == a
    @test LinearElasticity.σ_voigt(a) == A

    # stress tensor in three dimensions
    A = @SMatrix [ 1.0 2.0 3.0
                   2.0 4.0 5.0
                   3.0 5.0 6.0 ]
    a = @SVector [ 1.0; 4.0; 6.0; 5.0; 3.0; 2.0 ]
    @test LinearElasticity.σ_voigt(A) == a
    @test LinearElasticity.σ_voigt(a) == A

    # inner product
    σ = @SMatrix [ 1.0 2.0
                   2.0 3.0 ]
    ϵ = @SMatrix [ 4.0 6.0
                   6.0 5.0 ]
    σ̄ = LinearElasticity.σ_voigt(σ)
    ϵ̄ = LinearElasticity.ϵ_voigt(ϵ)
    @test dot(σ,ϵ) == dot(σ̄, ϵ̄)
end

@testset "Strain mapping in two dimensions" begin
    # parameteric domain [-2,3] × [4,7]
    Ω̂ = Interval(-2.0, 3.0) ⨱ Interval(4.0, 7.0)

    # geometric mapping F: (ξ,η) → (x,y)
    F = GeometricMapping(Ω̂, (ξ,η) -> ξ+0.1η^2, (ξ,η) -> η-0.1(ξ+2.5)^2)

    # Jacobian Jᵢⱼ = ∂ᵢ Fⱼ and its inverse
    J = (ξ,η) -> @SMatrix [1.0 -0.2*(ξ+2.5); 0.2η 1.0]
    J⁻¹ = (ξ,η) -> inv(J(ξ,η))

    # displacement and its gradient
    u = Field((ξ,η) -> ξ + 0.1η, (ξ,η) -> η + 0.1ξ)
    ∇u = (ξ,η) -> @SMatrix [ 1.0 0.1; 0.1 1.0 ]

    # symmetric strain tensor
    ϵ = (ξ,η) -> 0.5 * (J⁻¹(ξ,η) * ∇u(ξ,η) + (J⁻¹(ξ,η) * ∇u(ξ,η))')

    # Cartesian grid of test points
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    # strain evaluation at all x̂
    @evaluate strain = LinearElasticity.Strain(F,u)(x̂)

    # test at each grid point
    @test all(k -> strain[k] ≈ ϵ(x̂[k]...), eachindex(x̂))
    @test all(k -> strain[k] ≈ transpose(strain[k]), eachindex(x̂))
end

@testset "Strain mapping in three dimensions" begin
    # parameteric domain [-2,3] × [4,7] × [3,7]
    Ω̂ = Interval(-2.0, 3.0) ⨱ Interval(2.0, 7.0) ⨱ Interval(-1.0, 4.0)

    # geometric mapping F: (ξ,η) → (x,y)
    F = GeometricMapping(Ω̂,
        (ξ,η,ζ) -> ξ + 0.025(η^2 - ζ^2),
        (ξ,η,ζ) -> η + 0.025(ξ^2 - ζ^2),
        (ξ,η,ζ) -> ζ + 0.025(ξ^2 - η^2)
    )

    # Jacobian Jᵢⱼ = ∂ᵢFⱼ and its inverse
    J = (ξ,η,ζ) -> @SMatrix [
        1.0    0.05ξ  0.05ξ
        0.05η  1.0    -0.05η
        -0.05ζ -0.05ζ 1.0
    ]
    J⁻¹ = (ξ,η,ζ) -> inv(J(ξ,η,ζ))

    # displacement and its gradient
    u = Field(
        (ξ,η,ζ) -> ξ + 0.1(η+ζ),
        (ξ,η,ζ) -> η + 0.1(ξ+ζ),
        (ξ,η,ζ) -> ζ + 0.1(ξ+η)
    )
    ∇u = (ξ,η,ζ) -> @SMatrix [
        1.0 0.1 0.1
        0.1 1.0 0.1
        0.1 0.1 1.0
    ]

    # symmetric strain tensor
    ϵ = (ξ,η,ζ) -> 0.5 * (J⁻¹(ξ,η,ζ) * ∇u(ξ,η,ζ) + (J⁻¹(ξ,η,ζ) * ∇u(ξ,η,ζ))')

    # Cartesian grid of test points
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    ## strain evaluation at all x̂
    @evaluate strain = LinearElasticity.Strain(F,u)(x̂)

    ## test at each grid point
    @test all(k -> strain[k] ≈ ϵ(x̂[k]...), eachindex(x̂))
    @test all(k -> strain[k] ≈ transpose(strain[k]), eachindex(x̂))
end

@testset "Stress mapping in two dimensions (biaxial strain)" begin
    # parameteric domain [0,1] × [0,1]
    Ω̂ = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)

    # Cartesian grid of test points
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    # identity geometric mapping F: (ξ,η) → (x,y)
    F = GeometricMapping(Ω̂, (ξ,η) -> ξ, (ξ,η) -> η)

    # material mapping
    E = (ξ,η) -> 210e3
    ν = (ξ,η) -> 0.3
    material = LinearElasticity.PlaneStrain(E, ν)

    # displacement
    u = Field((ξ,η) -> ξ, (ξ,η) -> η)

    # strain tensor mapping
    strain = LinearElasticity.Strain(F,u)

    # Cauchy stress mapping
    stress = LinearElasticity.CauchyStress(strain, material)

    # von Mises stress mapping
    mises = LinearElasticity.VonMisesStress(stress)

    # stress evaluation at all x̂
    @evaluate σ = stress(x̂)
    @evaluate σᵥ = mises(x̂)

    # test data
    Id = I(2)
    λ = (ξ,η) -> E(ξ,η) * ν(ξ,η) / ( (1+ν(ξ,η))*(1-2ν(ξ,η)) )
    μ = (ξ,η) -> E(ξ,η)/(2(1+ν(ξ,η)))
    ϵ̄ = LinearElasticity.ϵ_voigt(@SVector [1.0; 1.0; 0.0])
    σ̄ = (ξ,η) -> λ(ξ,η) * Id * tr(ϵ̄) + 2μ(ξ,η) * ϵ̄
    s = (ξ,η) -> σ̄(ξ,η) - tr(σ̄(ξ,η))/3 * Id
    σ̄ᵥ = (ξ,η) -> sqrt(3/2 * dot(s(ξ,η), s(ξ,η)))

    # test at each grid point
    @test all(k -> σ[k] ≈ σ̄(x̂[k]...), eachindex(x̂))
    @test all(k -> σᵥ[k] ≈ σ̄ᵥ(x̂[k]...), eachindex(x̂))
    @test all(k -> σ[k] ≈ transpose(σ[k]), eachindex(x̂))
end

@testset "Stress mapping in two dimensions (pure shear strain)" begin
    # parameteric domain [0,1] × [0,1]
    Ω̂ = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)

    # Cartesian grid of test points
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    # identity geometric mapping F: (ξ,η) → (x,y)
    F = GeometricMapping(Ω̂, (ξ,η) -> ξ, (ξ,η) -> η)

    # material mapping
    E = (ξ,η) -> 210e3
    ν = (ξ,η) -> 0.3
    material = LinearElasticity.PlaneStrain(E, ν)

    # displacement
    u = Field((ξ,η) -> η, (ξ,η) -> ξ)

    # strain tensor mapping
    strain = LinearElasticity.Strain(F,u)

    # Cauchy stress mapping
    stress = LinearElasticity.CauchyStress(strain, material)

    # von Mises stress mapping
    mises = LinearElasticity.VonMisesStress(stress)

    # stress evaluation at all x̂
    @evaluate σ = stress(x̂)
    @evaluate σᵥ = mises(x̂)

    # test data
    ϵ̄ = @SVector [0.0; 0.0; 2.0]
    μ = (ξ,η) -> E(ξ,η)/(2(1+ν(ξ,η)))
    σ̄ = (ξ,η) -> 2μ(ξ,η) * LinearElasticity.ϵ_voigt(ϵ̄)
    σ̄ᵥ = (ξ,η) -> sqrt(3) * σ̄(ξ,η)[1,2] # ≠ 0.0

    # test at each grid point
    @test all(k -> σ[k] ≈ σ̄(x̂[k]...), eachindex(x̂))
    @test all(k -> σᵥ[k] ≈ σ̄ᵥ(x̂[k]...), eachindex(x̂))
    @test all(k -> σ[k] ≈ transpose(σ[k]), eachindex(x̂))
end

@testset "Stress mapping in three dimensions (triaxial strain)" begin
    # parameteric domain [0,1] × [0,1]
    Ω̂ = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)

    # Cartesian grid of test points
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    # identity geometric mapping F: (ξ,η,ζ) → (x,y,z)
    F = GeometricMapping(Ω̂, (ξ,η,ζ) -> ξ, (ξ,η,ζ) -> η, (ξ,η,ζ) -> ζ)

    # material mapping
    E = (ξ,η,ζ) -> 210e3
    ν = (ξ,η,ζ) -> 0.3
    material = LinearElasticity.Isotropic(E, ν)

    # displacement
    u = Field((ξ,η,ζ) -> ξ, (ξ,η,ζ) -> η, (ξ,η,ζ) -> ζ)

    # strain tensor mapping
    strain = LinearElasticity.Strain(F,u)

    # Cauchy stress mapping
    stress = LinearElasticity.CauchyStress(strain, material)

    # von Mises stress mapping
    mises = LinearElasticity.VonMisesStress(stress)

    # stress evaluation at all x̂
    @evaluate σ = stress(x̂)
    @evaluate σᵥ = mises(x̂)

    # test data
    Id = I(3)
    λ = (ξ,η,ζ) -> E(ξ,η,ζ) * ν(ξ,η,ζ) / ( (1+ν(ξ,η,ζ))*(1-2ν(ξ,η,ζ)) )
    μ = (ξ,η,ζ) -> E(ξ,η,ζ)/(2(1+ν(ξ,η,ζ)))
    ϵ̄ = LinearElasticity.ϵ_voigt(@SVector [1.0; 1.0; 1.0; 0.0; 0.0; 0.0])
    σ̄ = (ξ,η,ζ) -> λ(ξ,η,ζ) * Id * tr(ϵ̄) + 2μ(ξ,η,ζ) * ϵ̄
    σ̄ᵥ = (ξ,η,ζ) -> 0.0

    # test at each grid point
    @test all(k -> σ[k] ≈ σ̄(x̂[k]...), eachindex(x̂))
    @test all(k -> σᵥ[k] ≈ σ̄ᵥ(x̂[k]...), eachindex(x̂))
    @test all(k -> σ[k] ≈ transpose(σ[k]), eachindex(x̂))
end

@testset "Stress mapping in three dimensions (pure shear strain)" begin
    # parameteric domain [0,1] × [0,1]
    Ω̂ = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)

    # Cartesian grid of test points
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    # identity geometric mapping F: (ξ,η,ζ) → (x,y,z)
    F = GeometricMapping(Ω̂, (ξ,η,ζ) -> ξ, (ξ,η,ζ) -> η, (ξ,η,ζ) -> ζ)

    # material mapping
    E = (ξ,η,ζ) -> 210e3
    ν = (ξ,η,ζ) -> 0.3
    material = LinearElasticity.Isotropic(E, ν)

    # displacement
    u = Field((ξ,η,ζ) -> η+ζ, (ξ,η,ζ) -> ξ+ζ, (ξ,η,ζ) -> η+ξ)

    # strain tensor mapping
    strain = LinearElasticity.Strain(F,u)

    # Cauchy stress mapping
    stress = LinearElasticity.CauchyStress(strain, material)

    # von Mises stress mapping
    mises = LinearElasticity.VonMisesStress(stress)

    # stress evaluation at all x̂
    @evaluate σ = stress(x̂)
    @evaluate σᵥ = mises(x̂)

    # test data
    Id = I(3)
    λ = (ξ,η,ζ) -> E(ξ,η,ζ) * ν(ξ,η,ζ) / ( (1+ν(ξ,η,ζ))*(1-2ν(ξ,η,ζ)) )
    μ = (ξ,η,ζ) -> E(ξ,η,ζ)/(2(1+ν(ξ,η,ζ)))
    ϵ̄ = LinearElasticity.ϵ_voigt(@SVector [0.0; 0.0; 0.0; 2.0; 2.0; 2.0])
    σ̄ = (ξ,η,ζ) -> λ(ξ,η,ζ) * Id * tr(ϵ̄) + 2μ(ξ,η,ζ) * ϵ̄
    s = (ξ,η,ζ) -> σ̄(ξ,η,ζ) - tr(σ̄(ξ,η,ζ))/3 * Id
    σ̄ᵥ = (ξ,η,ζ) -> sqrt(3/2 * dot(s(ξ,η,ζ), s(ξ,η,ζ)))

    # test at each grid point
    @test all(k -> σ[k] ≈ σ̄(x̂[k]...), eachindex(x̂))
    @test all(k -> σᵥ[k] ≈ σ̄ᵥ(x̂[k]...), eachindex(x̂))
    @test all(k -> σ[k] ≈ transpose(σ[k]), eachindex(x̂))
end

@testset "Plane stress material matrix" begin
    E = (ξ,η) -> 210e3
    ν = (ξ,η) -> 0.3
    C̄ = (ξ,η) -> begin
        local E = 210e3
        local ν = 0.3
        E/(1-ν^2) * [1  ν  0
                     ν  1  0
                     0  0  (1-ν)/2]
    end
    
    # plane stress material mapping
    material = LinearElasticity.PlaneStress(E, ν)

    # test grid points
    Ω̂ = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    # identity geometric mapping F: (ξ,η) → (x,y)
    F = GeometricMapping(Ω̂, (ξ,η) -> ξ, (ξ,η) -> η)
    @evaluate x = F(x̂)

    # evaluation at Cartesian grid
    @evaluate C = material(x̂)
    @test all(k -> C[k] ≈ C̄(x̂[k]...), eachindex(x̂))

    # evaluation at EvaluationSet
    @evaluate C = material(x)
    @test all(k -> C[k] ≈ C̄(x[k]...), eachindex(x))
end

@testset "Plane strain material matrix" begin
    E = (ξ,η) -> 210e3
    ν = (ξ,η) -> 0.3
    C̄ = (ξ,η) -> begin
        local E = 210e3
        local ν = 0.3
        E/((1+ν)*((1-2ν))) * [1-ν  ν    0
                              ν    1-ν  0
                              0    0    (1-2ν)/2]
    end
    
    # plane strain material mapping
    material = LinearElasticity.PlaneStrain(E, ν)

    # test grid points
    Ω̂ = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    # identity geometric mapping F: (ξ,η) → (x,y)
    F = GeometricMapping(Ω̂, (ξ,η) -> ξ, (ξ,η) -> η)
    @evaluate x = F(x̂)

    # evaluation at Cartesian grid
    @evaluate C = material(x̂)
    @test all(k -> C[k] ≈ C̄(x̂[k]...), eachindex(x̂))

    # evaluation at EvaluationSet
    @evaluate C = material(x)
    @test all(k -> C[k] ≈ C̄(x[k]...), eachindex(x))
end

@testset "Isotropic material matrix" begin
    E = (ξ,η,ζ) -> 210e3
    ν = (ξ,η,ζ) -> 0.3
    C̄ = (ξ,η,ζ) -> begin
        local E = 210e3     
        local ν = 0.3
        E/((1+ν)*((1-2ν))) * [1-ν  ν    ν    0         0         0
                              ν    1-ν  ν    0         0         0
                              ν    ν    1-ν  0         0         0
                              0    0    0    (1-2ν)/2  0         0
                              0    0    0    0         (1-2ν)/2  0 
                              0    0    0    0         0         (1-2ν)/2 ]
    end
    
    # isotropic material mapping
    material = LinearElasticity.Isotropic(E, ν)

    # test grid points
    Ω̂ = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    # identity geometric mapping F: (ξ,η,ζ) → (x,y,z)
    F = GeometricMapping(Ω̂, (ξ,η,ζ) -> ξ, (ξ,η,ζ) -> η, (ξ,η,ζ) -> ζ)
    @evaluate x = F(x̂)

    # evaluation at Cartesian grid
    @evaluate C = material(x̂)
    @test all(k -> C[k] ≈ C̄(x̂[k]...), eachindex(x̂))

    # evaluation at EvaluationSet
    @evaluate C = material(x)
    @test all(k -> C[k] ≈ C̄(x[k]...), eachindex(x))
end

@testset "Orthotropic material matrix" begin
    E₁ = (x,y,z) -> 13.810e9
    E₂ = (x,y,z) -> 0.678e9
    E₃ = (x,y,z) -> 1.311e9
    ν₂₃ = (x,y,z) -> 0.420
    ν₁₃ = (x,y,z) -> 0.460
    ν₁₂ = (x,y,z) -> 0.500
    G₂₃ = (x,y,z) -> 0.255e9
    G₁₃ = (x,y,z) -> 1.013e9
    G₁₂ = (x,y,z) -> 0.753e9
    ν₃₂ = (x,y,z) -> ν₂₃(x,y,z) / E₂(x,y,z) * E₃(x,y,z)
    ν₃₁ = (x,y,z) -> ν₁₃(x,y,z) / E₁(x,y,z) * E₃(x,y,z)
    ν₂₁ = (x,y,z) -> ν₁₂(x,y,z) / E₁(x,y,z) * E₂(x,y,z)
    C̄ = (ξ,η,ζ) -> begin
        local E₁  = 13.810e9
        local E₂  = 0.678e9
        local E₃  = 1.311e9
        local ν₂₃ = 0.420
        local ν₁₃ = 0.460
        local ν₁₂ = 0.500
        local G₂₃ = 0.255e9
        local G₁₃ = 1.013e9
        local G₁₂ = 0.753e9
        local ν₃₂ = ν₂₃ / E₂ * E₃
        local ν₃₁ = ν₁₃ / E₁ * E₃
        local ν₂₁ = ν₁₂ / E₁ * E₂
        D = 1 - ν₁₂ * ν₂₁ - ν₁₃ * ν₃₁ - ν₂₃ * ν₃₂ - 2 * ν₁₂ * ν₂₃ * ν₃₁
        [(1 - ν₂₃ * ν₃₂) / D * E₁     (ν₂₁ + ν₂₃ * ν₃₁) / D * E₁     (ν₃₁ + ν₃₂ * ν₂₁) / D * E₁     0    0    0
         (ν₁₂ + ν₁₃ * ν₃₂) / D * E₂   (1 - ν₁₃ * ν₃₁) / D * E₂       (ν₃₂ + ν₃₁ * ν₁₂) / D * E₂     0    0    0
         (ν₁₃ + ν₁₂ * ν₂₃) / D * E₃   (ν₂₃ + ν₂₁ * ν₁₃) / D * E₃     (1 - ν₁₂ * ν₂₁) / D * E₃       0    0    0
         0                            0                              0                              G₂₃  0    0
         0                            0                              0                              0    G₁₃  0
         0                            0                              0                              0    0    G₁₂]
    end
    
    # orthotropic material mapping
    material = LinearElasticity.Orthotropic(E₁, E₂, E₃, ν₂₃, ν₃₂, ν₁₃, ν₃₁, ν₁₂, ν₂₁, G₂₃, G₁₃, G₁₂)

    # test grid points
    Ω̂ = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    # identity geometric mapping F: (ξ,η,ζ) → (x,y,z)
    F = GeometricMapping(Ω̂, (ξ,η,ζ) -> ξ, (ξ,η,ζ) -> η, (ξ,η,ζ) -> ζ)
    @evaluate x = F(x̂)

    # evaluation at Cartesian grid
    @evaluate C = material(x̂)
    @test all(k -> C[k] ≈ C̄(x̂[k]...), eachindex(x̂))

    # evaluation at EvaluationSet
    @evaluate C = material(x)
    @test all(k -> C[k] ≈ C̄(x[k]...), eachindex(x))
end

@testset "Pullback bilinear form in two dimensions" begin
    # parameteric domain [-2,3] × [4,7]
    Ω̂ = Interval(-2.0, 3.0) ⨱ Interval(4.0, 7.0)

    # geometric mapping F: (ξ,η) → (x,y)
    F = GeometricMapping(Ω̂, (ξ,η) -> ξ+0.1η^2, (ξ,η) -> η-0.1(ξ+2.5)^2)

    # material
    E = (ξ,η) -> 210e3
    ν = (ξ,η) -> 0.3
    material = LinearElasticity.PlaneStress(E, ν)
    C = (ξ,η) -> begin
        local E = 210e3
        local ν = 0.3
        E/(1-ν^2) * [1  ν  0
                     ν  1  0
                     0  0  (1-ν)/2]
    end

    # Jacobian Jᵢⱼ = ∂ᵢ Fⱼ and its inverse
    J = (ξ,η) -> @SMatrix [1.0 -0.2*(ξ+2.5); 0.2η 1.0]
    J⁻¹ = (ξ,η) -> inv(J(ξ,η))
    detJ = (ξ,η) -> det(J(ξ,η))

    # B-matrix
    B = LinearElasticity.BMatrix{2}()

    # test grid points
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    # test pullback
    test_data = (s,t,ξ,η) -> (J⁻¹(ξ,η)' * B[s]' * C(ξ,η) * B[t] * J⁻¹(ξ,η) * det(J(ξ,η)))
    pullback = LinearElasticity.PullbackBilinearForm(F, material)
    @evaluate data = pullback(x̂)
    for s in 1:2
        for t in 1:2
            @test all(k -> data[s,t][k] ≈ test_data(s,t,x̂[k]...), eachindex(x̂))
        end
    end
end

@testset "Pullback bilinear form in three dimensions" begin
    # parameteric domain [-2,3] × [2,7] × [-1,4]
    Ω̂ = Interval(-2.0, 3.0) ⨱ Interval(2.0, 7.0) ⨱ Interval(-1.0, 4.0)

    # geometric mapping F: (ξ,η) → (x,y)
    F = GeometricMapping(Ω̂,
        (ξ,η,ζ) -> ξ + 0.025(η^2 - ζ^2),
        (ξ,η,ζ) -> η + 0.025(ξ^2 - ζ^2),
        (ξ,η,ζ) -> ζ + 0.025(ξ^2 - η^2)
    )
    
    # material
    E = (ξ,η,ζ) -> 210e3
    ν = (ξ,η,ζ) -> 0.3
    C = (ξ,η,ζ) -> begin
        local E = 210e3     
        local ν = 0.3
        E/((1+ν)*((1-2ν))) * [1-ν  ν    ν    0         0         0
                              ν    1-ν  ν    0         0         0
                              ν    ν    1-ν  0         0         0
                              0    0    0    (1-2ν)/2  0         0
                              0    0    0    0         (1-2ν)/2  0 
                              0    0    0    0         0         (1-2ν)/2 ]
    end
    material = LinearElasticity.Isotropic(E, ν)

    # Jacobian Jᵢⱼ = ∂ᵢFⱼ and its inverse
    J = (ξ,η,ζ) -> @SMatrix [
        1.0    0.05ξ  0.05ξ
        0.05η  1.0    -0.05η
        -0.05ζ -0.05ζ 1.0
    ]
    J⁻¹ = (ξ,η,ζ) -> inv(J(ξ,η,ζ))
    detJ = (ξ,η) -> det(J(ξ,η))

    # B-matrix
    B = LinearElasticity.BMatrix{3}()

    # test grid points
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    # test pullback
    test_data = (s,t,ξ,η,ζ) -> (J⁻¹(ξ,η,ζ)' * B[s]' * C(ξ,η,ζ) * B[t] * J⁻¹(ξ,η,ζ) * det(J(ξ,η,ζ)))
    pullback = LinearElasticity.PullbackBilinearForm(F, material)
    @evaluate data = pullback(x̂)
    for s in 1:3
        for t in 1:3
            @test all(k -> data[s,t][k] ≈ test_data(s,t,x̂[k]...), eachindex(x̂))
        end
    end
end

@testset "Pullback boundary linear form in two dimensions" begin
    # parameteric domain [1.0,2.0] × [0.0,π/2]
    Ω̂ = Interval(1.0, 2.0) ⨱ Interval(0.0, π/2)

    # geometric mapping F: (r,θ) → (x,y)
    F = GeometricMapping(Ω̂, (r,θ) -> r*cos(θ), (r,θ) -> r*sin(θ))

    # Jacobian Jᵢⱼ = ∂ᵢ Fⱼ and its inverse
    J = (r,θ) -> [cos(θ) sin(θ); -r*sin(θ) r*cos(θ)]
    J⁻¹ = (r,θ) -> inv(J(r,θ))

    # normal in parametric space
    Normal = side -> begin
        if side == 1
            N = @SVector [-1.0; 0.0]
        elseif side == 2
            N = @SVector [1.0; 0.0]
        elseif side == 3
            N = @SVector [0.0; -1.0]
        elseif side == 4
            N = @SVector [0.0; 1.0]
        end
    end

    # oriented area in physical space
    normal = (ξ, side) -> begin
        if side == 1
            n = J⁻¹(1.0, ξ) * Normal(side) * det(J(1.0, ξ))
        elseif side == 2
            n = J⁻¹(2.0, ξ) * Normal(side) * det(J(2.0, ξ))
        elseif side == 3
            n = J⁻¹(ξ, 0.0) * Normal(side) * det(J(ξ, 0.0))
        elseif side == 4
            n = J⁻¹(ξ, π/2) * Normal(side) * det(J(ξ, π/2))
        end
    end

    # traction
    t = (x, y) -> @SMatrix [1.0 2.0; 3.0 4.0];

    # test grid points
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 7), Ω̂)

    # test pullback
    for side in 1:4
        pullback = LinearElasticity.PullbackBoundaryLinearForm(F, t, side)

        ∂x̂ = boundary(x̂, side)
        ∂F = boundary(F, side)

        @evaluate x = ∂F(∂x̂)
        @evaluate data = pullback(CartesianProduct(∂x̂, [0.0]))

        @test all(k -> all(data[k] .≈ t(x[k]...) * normal(∂x̂[k], side)), eachindex(∂x̂))
    end
end

@testset "Pullback boundary linear form in three dimensions" begin
    # parameteric domain [1.0,2.0] × [π/4,3π/4] × [0.0,π/2]
    Ω̂ = Interval(1.0, 2.0) ⨱ Interval(pi/4, 3π/4) ⨱ Interval(0.0, π/2)

    # geometric mapping F: (r,θ,ϕ) → (x,y,z)
    F = GeometricMapping(Ω̂,
        (r,θ,ϕ) -> r * sin(θ) * cos(ϕ),
        (r,θ,ϕ) -> r * sin(θ) * sin(ϕ),
        (r,θ,ϕ) -> r * cos(θ)
    )

    # Jacobian Jᵢⱼ = ∂ᵢ Fⱼ and its inverse
    J = (r,θ,ϕ) -> [
         sin(θ)*cos(ϕ)        sin(θ)*sin(ϕ)        cos(θ)
         r * cos(θ) * cos(ϕ)  r * cos(θ) * sin(ϕ)  -r * sin(θ)
        -r * sin(θ) * sin(ϕ)  r * sin(θ) * cos(ϕ)  0.0
    ]
    J⁻¹ = (r,θ,ϕ) -> inv(J(r,θ,ϕ))

    # normal in parametric space
    Normal = side -> begin
        if side == 1
            N = @SVector [-1.0; 0.0; 0.0]
        elseif side == 2
            N = @SVector [1.0; 0.0; 0.0]
        elseif side == 3
            N = @SVector [0.0; -1.0; 0.0]
        elseif side == 4
            N = @SVector [0.0; 1.0; 0.0]
        elseif side == 5
            N = @SVector [0.0; 0.0; -1.0]
        elseif side == 6
            N = @SVector [0.0; 0.0; 1.0]
        end
    end

    # oriented area in physical space
    normal = (ξ, η, side) -> begin
        if side == 1
            n = J⁻¹(1.0, ξ, η) * Normal(side) * det(J(1.0, ξ, η))
        elseif side == 2
            n = J⁻¹(2.0, ξ, η) * Normal(side) * det(J(2.0, ξ, η))
        elseif side == 3
            n = J⁻¹(ξ, π/4, η) * Normal(side) * det(J(ξ, π/4, η))
        elseif side == 4
            n = J⁻¹(ξ, 3π/4, η) * Normal(side) * det(J(ξ, 3π/4, η))
        elseif side == 5
            n = J⁻¹(ξ, η, 0.0) * Normal(side) * det(J(ξ, η, 0.0))
        elseif side == 6
            n = J⁻¹(ξ, η, π/2) * Normal(side) * det(J(ξ, η, π/2))
        end
    end

    # traction
    t = (x, y, z) -> @SMatrix [1.0 2.0 3.0
                               4.0 5.0 6.0
                               7.0 8.0 9.0];

    # test grid points
    x̂ = CartesianProduct(i -> IncreasingRange(i..., 3), Ω̂)

    # test pullback
    for side in 1:6
        pullback = LinearElasticity.PullbackBoundaryLinearForm(F, t, side)

        ∂x̂ = boundary(x̂, side)
        ∂F = boundary(F, side)

        @evaluate x = ∂F(∂x̂)
        @evaluate data = pullback(CartesianProduct(∂x̂..., [0.0]))

        @test all(k -> all(data[k] .≈ t(x[k]...) * normal(∂x̂[k]..., side)), eachindex(∂x̂))
    end
end

@testset "Hole in a two-dimensional plate" begin
    E, ν = 210e3, 0.3
    benchmark = LinearElasticity.benchmark_hole_in_plate_2d(; E=E, ν=ν)

    F = hole_in_square_plate()
    F = refine(F, method=pRefinement(1))
    F = refine(F, method=hRefinement(8))

    Dim = dimension(F)
    Ω = domain(F)

    #C₁ = ScalarFunctionSpaceConstraint{Dim}()
    #C₂ = ScalarFunctionSpaceConstraint{Dim}()
    #left_constraint!(C₁, 1; dim=1);
    #right_constraint!(C₂, 1; dim=1);
    #C = VectorFunctionSpaceConstraint(C₁, C₂)
    C = VectorSplineSpaceConstraints{Dim}()
    clamped_constraint!(C[1], :left)
    clamped_constraint!(C[2], :right)

    S = VectorSplineSpace(ScalarSplineSpace(F.space))

    uʰ = Field(S)
    ūʰ = Field(S)
    uₐ = Field((x,y) -> benchmark.displacement(x,y)[1], (x,y) -> benchmark.displacement(x,y)[2])

    σₐ = Field{Dim,Dim}(
        (x,y) -> benchmark.stress(x,y)[1,1],
        (x,y) -> benchmark.stress(x,y)[2,1],
        (x,y) -> benchmark.stress(x,y)[1,2],
        (x,y) -> benchmark.stress(x,y)[2,2]
    )

    ū = Field((x, y) -> 0.0, (x, y) -> 0.0)
    project!(ūʰ, onto=ū, method=Interpolation)

    t = benchmark.stress
    material = LinearElasticity.PlaneStrain((x,y) -> E, (x,y) -> ν)

    model = LinearElasticity.Model(F, S, C, uʰ, ūʰ, t, material)
    L = linear_operator(model; show_progress=false)
    b = forcing(L, model)

    P = FastDiagonalization(model)
    solver = TaigaPCG(L, P; itmax=1000, atol=1.0e-8, rtol=0.0)
    x₀, stats = linsolve!(solver, b)
    @test stats.niter < 76
    @test stats.residual < 10e-9

    uʰ = Field(S)
    x = apply_particular_solution(L, model, x₀)
    setcoeffs!(uʰ, S, x)

    ε = Taiga.LinearElasticity.Strain(F, uʰ)
    σ = Taiga.LinearElasticity.CauchyStress(ε, material)

    # check L₂ errors
    L₂ = l2_error(uʰ, to=uₐ ∘ F, relative=true)
    @test all(L₂ .< 10e-4)

    L₂ = l2_error(σ, to=σₐ ∘ F, relative=true)
    @test all(L₂ .< 10e-2)
end

@testset "Hole in a three-dimensional plate" begin
    E, ν = 210e3, 0.3
    benchmark = LinearElasticity.benchmark_hole_in_plate_3d(; E=E, ν=ν)
    
    Ω = Interval(1.0,4.0) ⨱ Interval(0.0,pi/2) ⨱ Interval(0.0,1.0)
    F = GeometricMapping(Ω, (r,θ,z) -> r*cos(θ), (r,θ,z) -> r*sin(θ), (r,θ,z) -> z)
    Dim = dimension(F)
    Δ = Partition(Ω, ntuple(d -> 6, Dim))

    uₐ = Field(
        (x,y,z) -> benchmark.displacement(x,y,z)[1],
        (x,y,z) -> benchmark.displacement(x,y,z)[2],
        (x,y,z) -> benchmark.displacement(x,y,z)[3]
    )

    σₐ = Field{Dim,Dim}(
        (x,y,z) -> benchmark.stress(x,y,z)[1,1],
        (x,y,z) -> benchmark.stress(x,y,z)[2,1],
        (x,y,z) -> benchmark.stress(x,y,z)[3,1],
        (x,y,z) -> benchmark.stress(x,y,z)[1,2],
        (x,y,z) -> benchmark.stress(x,y,z)[2,2],
        (x,y,z) -> benchmark.stress(x,y,z)[3,2],
        (x,y,z) -> benchmark.stress(x,y,z)[1,3],
        (x,y,z) -> benchmark.stress(x,y,z)[2,3],
        (x,y,z) -> benchmark.stress(x,y,z)[3,3],
    )

    S = VectorSplineSpace(ScalarSplineSpace(ntuple(d -> 4, Dim), Δ))
    C = VectorSplineSpaceConstraints{Dim}()
    clamped_constraint!(C[1], :top)
    clamped_constraint!(C[2], :bottom)
    clamped_constraint!(C[3], :back, :front)

    uʰ = Field(S)
    ūʰ = Field(S)
    ū = Field((x, y, z) -> 0.0, (x, y, z) -> 0.0, (x,y,z) -> 0.0)
    project!(ūʰ, onto=ū, method=Interpolation)

    t = benchmark.stress
    material = LinearElasticity.Isotropic((x,y,z) -> E, (x,y,z) -> ν)

    model = LinearElasticity.Model(F, S, C, uʰ, ūʰ, t, material)
    L = linear_operator(model; show_progress=false)
    b = forcing(L, model)
    x₀ = sparse(L) \ b

    x = apply_particular_solution(L, model, x₀)
    setcoeffs!(uʰ, S, x)

    ϵ = Taiga.LinearElasticity.Strain(F, uʰ)
    σ = Taiga.LinearElasticity.CauchyStress(ϵ, material)

    # check L₂ errors
    L₂ = l2_error(uʰ, to=uₐ ∘ F, relative=false)
    @test all(L₂ .< 10e-8)

    L₂ = l2_error(σ, to=σₐ ∘ F, relative=false)
    @test all(L₂ .< 10e-2)
end

@testset "Spherical cavity" begin
    E, ν = 210e3, 0.3
    benchmark = LinearElasticity.benchmark_spherical_cavity(; E=E, ν=ν)

    Ω = Interval(1.0,5.0) ⨱ Interval(0.0,pi/2) ⨱ Interval(0.0,pi/2)
    F = GeometricMapping(Ω, (r,β,θ) -> r*sin(β)*cos(θ), (r,β,θ) -> r*sin(β)*sin(θ), (r,β,θ) -> r*cos(β))
    Dim = dimension(F)
    Δ = Partition(Ω, ntuple(d -> 5, Dim))

    uₐ = Field(
        (x,y,z) -> benchmark.displacement(x,y,z)[1],
        (x,y,z) -> benchmark.displacement(x,y,z)[2],
        (x,y,z) -> benchmark.displacement(x,y,z)[3]
    )

    σₐ = Field{Dim,Dim}(
        (x,y,z) -> benchmark.stress(x,y,z)[1,1],
        (x,y,z) -> benchmark.stress(x,y,z)[2,1],
        (x,y,z) -> benchmark.stress(x,y,z)[3,1],
        (x,y,z) -> benchmark.stress(x,y,z)[1,2],
        (x,y,z) -> benchmark.stress(x,y,z)[2,2],
        (x,y,z) -> benchmark.stress(x,y,z)[3,2],
        (x,y,z) -> benchmark.stress(x,y,z)[1,3],
        (x,y,z) -> benchmark.stress(x,y,z)[2,3],
        (x,y,z) -> benchmark.stress(x,y,z)[3,3],
    )

    S = VectorSplineSpace(ScalarSplineSpace(ntuple(d -> 5, Dim), Δ))
    C = VectorSplineSpaceConstraints{Dim}()
    clamped_constraint!(C[1], :bottom, :front)
    clamped_constraint!(C[2], :bottom, :back)
    clamped_constraint!(C[3], :top)

    uʰ = Field(S)
    ūʰ = Field(S)

    ū = Field((x, y, z) -> 0.0, (x, y, z) -> 0.0, (x,y,z) -> 0.0)
    project!(ūʰ, onto=ū, method=Interpolation)

    t = benchmark.stress
    material = LinearElasticity.Isotropic((x,y,z) -> E, (x,y,z) -> ν)

    model = LinearElasticity.Model(F, S, C, uʰ, ūʰ, t, material)
    L = linear_operator(model; show_progress=false)
    b = forcing(L, model)
    x₀ = sparse(L) \ b

    x = apply_particular_solution(L, model, x₀)
    setcoeffs!(uʰ, S, x)

    ϵ = Taiga.LinearElasticity.Strain(F, uʰ)
    σ = Taiga.LinearElasticity.CauchyStress(ϵ, material)

    # check L₂ errors
    L₂ = l2_error(uʰ, to=uₐ ∘ F, relative=true)
    @test all(L₂ .< 10e-3)

    L₂ = l2_error(σ, to=σₐ ∘ F, relative=true)
    @test all(L₂ .< 0.15)
end