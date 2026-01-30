using Taiga, NURBS, StaticArrays, UnivariateSplines

@testset "Potential flow around cylinder" begin
    F = hole_in_square_plate()
    F = refine(F, method=pRefinement(3))
    F = refine(F, method=hRefinement(32))

    Dim = dimension(F)
    Ω = domain(F)

    C = ScalarSplineSpaceConstraints{Dim}()
    left_constraint!(C; dim=1);
    right_constraint!(C; dim=2);

    S = ScalarSplineSpace(F.space)

    uʰ = Field(S)
    ūʰ = Field(S)

    r(x, y) = abs(x + im * y)
    θ(x, y) = angle(x + im * y)

    U, R, ρ, p∞ = 1.0, 1.0, 1.0, 1.0;
    uₐ = ScalarFunction((x, y) -> U*r(x,y) * (1 + R^2 / r(x,y)^2) * cos(θ(x,y)))
    pₐ = ScalarFunction((x, y) -> 0.5 * ρ * U^2 * (2 * R^2/r(x,y)^2 * cos(2θ(x,y)) - R^4/r(x,y)^4) + p∞)
    vₐ = Field((x,y) -> U + U * R * (y^2 - x^2) / (x^2 + y^2)^2, (x,y) -> -2 * U * R^2 * x * y / (x^2 + y^2)^2 )
    t = (x, y) -> @SVector [0.0; 0.0] # traction
    project!(ūʰ, onto=Field(uₐ ∘ F), method=Interpolation)

    model = PotentialFlow(F, S, C, uʰ, ūʰ, t);
    L = linear_operator(model)
    b = forcing(L, model)

    P = FastDiagonalization(model)
    solver = TaigaPCG(L, P; atol=10e-12, rtol=0, itmax=100, history=true)
    x, stats = linsolve!(solver, b)

    x = apply_particular_solution(L, model, x)
    setcoeffs!(uʰ, S, x)

    vʰ = PotentialFlowModule.Velocity(F, uʰ)
    pʰ = PotentialFlowModule.Pressure(vʰ, U; ρ=ρ, p=p∞)

    L₂ = l2_error(uʰ, to=uₐ ∘ F, relative=true)[1]
    @test L₂ < 10e-10
    
    L₂ = l2_error(pʰ, to=pₐ ∘ F, relative=true)[1]
    @test L₂ < 10e-8

    L₂ = l2_error(vʰ, to=vₐ ∘ F, relative=true)
    @test all(L₂ .< 10e-6)
end

@testset "Potential flow on quarter annulus (prescribed traction)" begin
    F = annulus()

    F = refine(F, method=pRefinement(1))
    F = refine(F, method=hRefinement(32))

    Dim = dimension(F)
    Ω = domain(F)

    C = ScalarSplineSpaceConstraints{Dim}()
    left_constraint!(C; dim=1);
    S = ScalarSplineSpace(F.space)

    uʰ = Field(S)
    ūʰ = Field(S)

    r(x, y) = abs(x + im * y)
    θ(x, y) = angle(x + im * y)
    uₐ = ScalarFunction((x, y) -> 2 * (1 + θ(x,y) / π))
    t = (x, y) -> begin
        if θ(x,y) >= π/2
            @SVector [-2 * y / (π * (x^2 + y^2)); 2 * x / (π * (x^2 + y^2))]
        else
            @SVector [0; 0]
        end
    end

    project!(ūʰ, onto=Field(uₐ ∘ F), method=Interpolation)

    model = PotentialFlow(F, S, C, uʰ, ūʰ, t);
    L = linear_operator(model)
    b = forcing(L, model)

    P = FastDiagonalization(model)
    solver = TaigaPCG(L, P; atol=10e-12, rtol=0, itmax=100, history=true)
    x, stats = linsolve!(solver, b)

    x = apply_particular_solution(L, model, x)
    setcoeffs!(uʰ, S, x)

    L₂ = l2_error(uʰ, to=uₐ ∘ F, relative=true)[1]
    @test L₂ < 10e-10
end