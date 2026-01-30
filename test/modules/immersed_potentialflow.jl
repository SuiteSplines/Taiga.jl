using Taiga, NURBS, StaticArrays, UnivariateSplines, Algoim, IgaBase, LinearAlgebra
import SparseArrays: sparse
using ForwardDiff

import Taiga.ImmersedPotentialFlowModule: StreamFunction

@testset "Potential flow around cylinder" begin
    r(x, y) = abs(x + im * y)
    θ(x, y) = angle(x + im * y)
    U, R, ρ, p∞ = 1.0, 1.0, 1.0, 1.0
    uₐ = ScalarFunction((x, y) -> U*r(x,y) * (1 + R^2 / r(x,y)^2) * cos(θ(x,y)))
    pₐ = ScalarFunction((x, y) -> 0.5 * ρ * U^2 * (2 * R^2/r(x,y)^2 * cos(2θ(x,y)) - R^4/r(x,y)^4) + p∞)
    vₐ = Field((x,y) -> U + U * R * (y^2 - x^2) / (x^2 + y^2)^2, (x,y) -> -2 * U * R^2 * x * y / (x^2 + y^2)^2 )

    F = ImmersedPotentialFlowModule.embedding(8.0, 8.0)
    F = refine(F, method=pRefinement(3))
    F = refine(F, method=hRefinement(32))
    ϕ = AlgoimCallLevelSetFunction(
        x -> -x[1]*x[1] - x[2]*x[2] + R^2,  # ϕ
        x -> [-2.0*x[1], -2.0*x[2]]         # ∇ϕ
    )

    Dim = dimension(F)
    Ω = domain(F)

    C = ScalarSplineSpaceConstraints{Dim}()
    left_constraint!(C; dim=1);
    right_constraint!(C; dim=1);
    left_constraint!(C; dim=2);
    right_constraint!(C; dim=2);

    S = ScalarSplineSpace(F.space)
    
    uʰ = Field(S)
    ūʰ = Field(S)

    ū = GeometricMapping(Ω, (x,y) -> uₐ(x,y))
    t = (x, y) -> @SVector [0.0; 0.0] # traction

    for side = 1:2Dim
        project!(boundary(ūʰ[1],side), onto=boundary(ū,side), method=Interpolation)
    end

    model = ImmersedPotentialFlow(F, S, C, uʰ, ūʰ, t, ϕ);
    L = linear_operator(model; show_progress=false)
    b = forcing(L, model)
    x = sparse(L) \ b

    x = apply_particular_solution(L, model, x)
    setcoeffs!(uʰ, S, x)

    velocity = ImmersedPotentialFlowModule.Velocity(F, uʰ)
    pressure = ImmersedPotentialFlowModule.Pressure(velocity, 1.0; ρ=1.0, p=1.0)

    pʰ = Field(S)
    vʰ = Field(VectorSplineSpace(S,S))

    project!(ϕ, pʰ; onto=pressure, method=GalerkinProjection)
    project!(ϕ, vʰ; onto=velocity, method=GalerkinProjection)

    # check errors norms
    L₂ = l2_error(ϕ, uʰ, to=uₐ, relative=true)[1]
    @test L₂ < 10e-5

    L₂ = l2_error(ϕ, pʰ, to=pₐ, relative=true)[1]
    @test L₂ < 10e-3

    L₂ = l2_error(ϕ, vʰ, to=vₐ, relative=true)
    @test L₂[1] < 10e-3
    @test L₂[2] < 10e-3
end

@testset "Potential flow around a rotating cylinder" begin
    r(x, y) = abs(x + im * y)
    θ(x, y) = angle(x + im * y)
    U, R, ρ, p∞, Γ = 1.0, 1.0, 1.0, 1.0, 6.0*pi;
    Ψ = ScalarFunction((x,y) -> -U * r(x,y) * sin(θ(x,y)) * (R^2/r(x,y)^2 - 1) + Γ/(2π) * log(r(x,y)/R))
    ∇Ψ = Field(
        (x, y) -> ForwardDiff.derivative(x -> Ψ.data(x,y), x),
        (x, y) -> ForwardDiff.derivative(y -> Ψ.data(x,y), y)
    )
    v = Field((x,y) -> ∇Ψ.data[2](x,y), (x,y) -> -∇Ψ.data[1](x,y))
    p = ScalarFunction((x, y) -> p∞ + 0.5*ρ * (U^2 - (v[1].data(x,y)^2 + v[2].data(x,y)^2) ))

    F = ImmersedPotentialFlowModule.embedding(8.0, 8.0)
    F = refine(F, method=pRefinement(1))
    F = refine(F, method=hRefinement(32))

    ϕ = AlgoimCallLevelSetFunction(
        x -> -x[1]*x[1] - x[2]*x[2] + R^2,  # ϕ
        x -> [-2.0*x[1], -2.0*x[2]]         # ∇ϕ
    )

    Dim = dimension(F)
    Ω = domain(F)

    C = ScalarSplineSpaceConstraints{Dim}()
    left_constraint!(C; dim=1);
    right_constraint!(C; dim=1);
    left_constraint!(C; dim=2);
    right_constraint!(C; dim=2);

    S = ScalarSplineSpace(F.space)
    Ψʰ = Field(S)
    Ψ̄ʰ = Field(S)

    t = (x, y) -> begin
        @SVector [ ∇Ψ[1].data(x,y), ∇Ψ[2].data(x,y) ] # traction
    end

    Ψ̄ = GeometricMapping(Ω, (x,y) -> Ψ(x,y))
    for side = 1:2Dim
        project!(boundary(Ψ̄ʰ[1],side), onto=boundary(Ψ̄,side), method=Interpolation)
    end

    model = ImmersedPotentialFlow(F, S, C, Ψʰ, Ψ̄ʰ, t, ϕ);
    L = linear_operator(model; show_progress=false)
    b = forcing(L, model)
    x = sparse(L) \ b
    x = apply_particular_solution(L, model, x)
    setcoeffs!(Ψʰ, S, x)

    velocity = ImmersedPotentialFlowModule.Velocity{StreamFunction}(F, Ψʰ)
    pressure = ImmersedPotentialFlowModule.Pressure(velocity, 1.0; ρ=1.0, p=1.0)

    pʰ = Field(S)
    vʰ = Field(VectorSplineSpace(S,S))

    project!(ϕ, pʰ; onto=pressure, method=GalerkinProjection)
    project!(ϕ, vʰ; onto=velocity, method=GalerkinProjection)

    L₂ = l2_error(ϕ, Ψʰ, to=Ψ, relative=true)[1]
    @test L₂ < 10e-5

    L₂ = l2_error(ϕ, vʰ, to=v, relative=true)
    @test L₂[1] < 10e-3
    @test L₂[2] < 10e-3

    L₂ = l2_error(ϕ, pʰ, to=p, relative=true)[1]
    @test L₂ < 10e-3
end

@testset "Potential flow through a torus (integration test only)" begin
    F = ImmersedPotentialFlowModule.embedding(12.0, 12.0, 12.0)
    F = refine(F, method=hRefinement(8))
    
    R₁, R₂ = 2.0, 1.0
    ϕ = AlgoimCallLevelSetFunction(
        x -> 4 * R₁*R₁ * (x[1]*x[1] + x[2]*x[2]) - (x[1]*x[1]  + x[2]*x[2] + x[3]*x[3] + R₁*R₁ - R₂*R₂) * (x[1]*x[1]  + x[2]*x[2] + x[3]*x[3] + R₁*R₁ - R₂*R₂),  # ϕ
        x -> [
            8 * R₁*R₁ * x[1] - 4*x[1]*(x[1]*x[1]  + x[2]*x[2] + x[3]*x[3] + R₁*R₁ - R₂*R₂),
            8 * R₁*R₁ * x[2] - 4*x[2]*(x[1]*x[1]  + x[2]*x[2] + x[3]*x[3] + R₁*R₁ - R₂*R₂),
            -4*x[3]*(x[1]*x[1]  + x[2]*x[2] + x[3]*x[3] + R₁*R₁ - R₂*R₂)
        ] # ∇ϕ
    )
    
    Dim = dimension(F)
    Ω = domain(F)
    
    C = ScalarSplineSpaceConstraints{Dim}()
    left_constraint!(C; dim=1);
    right_constraint!(C; dim=1);
    left_constraint!(C; dim=2);
    right_constraint!(C; dim=3);
    left_constraint!(C; dim=3);
    right_constraint!(C; dim=3);
    S = ScalarSplineSpace(F.space)
    
    uʰ = Field(S)
    ūʰ = Field(S)
    ū = GeometricMapping(Ω, (x,y,z) -> z)
    t = (x, y, z) -> @SVector [0.0; 0.0; 0.0]
    for side = 1:2Dim
        project!(boundary(ūʰ[1],side), onto=boundary(ū,side), method=Interpolation)
    end
    
    model = ImmersedPotentialFlow(F, S, C, uʰ, ūʰ, t, ϕ);
    L = linear_operator(model, show_progress=false)
    b = forcing(L, model)
    
    solver = TaigaCG(L, atol=10e-8, rtol=0, itmax=2500, history=true)
    x, stats = linsolve!(solver, b)
    
    x = apply_particular_solution(L, model, x)
    setcoeffs!(uʰ, S, x)
    
    velocity = ImmersedPotentialFlowModule.Velocity(F, uʰ)
    pressure = ImmersedPotentialFlowModule.Pressure(velocity, 1.0; ρ=1.0, p=1.0)

    pʰ = Field(S)
    vʰ = Field(VectorSplineSpace(S,S,S))

    project!(ϕ, pʰ; onto=pressure, method=GalerkinProjection)
    project!(ϕ, vʰ; onto=velocity, method=GalerkinProjection)

    # todo: reference solution + L₂ norm
end

@testset "Cartesian background mesh" begin
    for side_lengths in [(7,11), (7,11,13)]
        F = ImmersedPotentialFlowModule.embedding(side_lengths...)
        F = refine(F, method=hRefinement(3))
        x = CartesianProduct(breakpoints, F.space)

        pullback = ImmersedPotentialFlowModule.PullbackBody(F)
        @evaluate Y = pullback(x)

        Dim = dimension(F)
        Id = I(Dim)

        @test all(k -> Y[k] ≈ Id, eachindex(x))
    end
end