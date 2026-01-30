using Taiga
using AbstractMappings, NURBS, SpecialSpaces, IgaBase
using StaticArrays, SpecialFunctions, Krylov, LinearAlgebra

@testset "Poisson periodic annulus model" begin
    # define mapping
    F = annulus(; inner_radius=11.064709488501170, outer_radius=17.61596604980483, β=2π)
    F = refine(F, method=pRefinement(3))
    F = refine(F, method=hRefinement(25))

    # define dimension
    Dim = dimension(F)

    # get domain mapping is defined on
    Ω = domain(F)

    # define space constraints
    C = ScalarSplineSpaceConstraints{Dim}()
    periodic_constraint!(C; c=collect(1:F.space[1].p), dim=1);
    left_constraint!(C; dim=2);
    right_constraint!(C; dim=2);

    # define scalar spline space
    S = ScalarSplineSpace(F.space)

    # define solution field
    uʰ = Field(S)

    # define field to satisfy Dirichlet boundary conditions
    ūʰ = Field(S);

    # define transformation from Cartesian to polar coordinates
    r(x, y) = abs(x + im * y)
    θ(x, y) = angle(x + im * y)

    # define analytical solution
    uₐ = ScalarFunction((x, y) -> besselj(4, r(x, y)) * cos(4θ(x, y)) + 1)

    # define Poisson parameters
    f = (x, y) -> uₐ(x, y) - 1; # bodyforce (-Δuʰ = f)
    ū = (x, y) -> 1.0; # function satisfying boundary conditions
    κ = (x, y) -> SMatrix{2,2,Float64}(I) # conductivity
    t = (x, y) -> @SVector [0.0; 0.0] # traction

    # project ū onto solution space
    project!(ūʰ, onto=Field(ū), method=Interpolation)

    # initialize model
    model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t);

    # construct linear operator
    L = linear_operator(model)

    # construct rhs vector
    b = forcing(L)

    # construct FastDiagonalization preconditioner using modal splines data approximation
    p̃, ñₑ = 2, 3
    P₁ = FastDiagonalization(model; method=ModalSplines, spaces=ntuple(k -> SplineSpace(p̃, domain(F.space[k]), ñₑ), 2))
    P₂ = FastDiagonalization(model; method=Wachspress, niter=2)

    # compare forcing(model) to forcing!(b, model)
    b_test = zeros(size(L, 1))
    forcing!(b_test, L)
    @test all(b .== b_test)

    # solve
    x₀, stats = cg(L, b)
    niter_test = stats.niter

    # solve with FD preconditioner using ModalSplines
    x₀, stats = cg(L, b; M=P₁)
    @test stats.niter < niter_test # a very crude test

    # solve with FD preconditioner using Wachspress
    x₀, stats = cg(L, b; M=P₂)
    @test stats.niter < niter_test # a very crude test

    # apply particular solution
    x = apply_particular_solution(L, x₀)

    # apply solution to field coefficients
    setcoeffs!(uʰ, S, x)

    # compute L₂ error norm
    L₂ = l2_error(uʰ, to=uₐ ∘ F, relative=true)[1]
    @test L₂ < 10e-5
end

@testset "Poisson periodic annulus matrix-free model" begin
    # define mapping
    F = annulus(; inner_radius=11.064709488501170, outer_radius=17.61596604980483, β=2π)
    F = refine(F, method=pRefinement(1))
    F = refine(F, method=hRefinement(3))

    # define dimension
    Dim = dimension(F)

    # get domain mapping is defined on
    Ω = domain(F)

    # define space constraints
    C = ScalarSplineSpaceConstraints{Dim}()
    periodic_constraint!(C; c=collect(1:F.space[1].p), dim=1);
    left_constraint!(C; dim=2);
    right_constraint!(C; dim=2);

    # define scalar spline space
    S = ScalarSplineSpace(F.space)

    # define solution field
    uʰ = Field(S)

    # define field to satisfy Dirichlet boundary conditions
    ūʰ = Field(S);

    # define transformation from Cartesian to polar coordinates
    r(x, y) = abs(x + im * y)
    θ(x, y) = angle(x + im * y)

    # define analytical solution
    uₐ = ScalarFunction((x, y) -> besselj(4, r(x, y)) * cos(4θ(x, y)) + 1)

    # define Poisson parameters
    f = (x, y) -> uₐ(x, y) - 1; # bodyforce (-Δuʰ = f)
    ū = (x, y) -> 1.0; # function satisfying boundary conditions
    κ = (x, y) -> SMatrix{2,2,Float64}(I) # conductivity
    t = (x, y) -> @SVector [1.0; 1.0] # traction

    # project ū onto solution space
    project!(ūʰ, onto=Field(ū), method=Interpolation)

    # initialize model
    model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t);

    # construct linear operator
    L = linear_operator(model; matrixfree=true)
    L_test = linear_operator(model)
    @test norm(Matrix(L) - Matrix(L_test)) < 10e-13
    
    # construct rhs vector
    b = forcing(model)
    b_test = forcing(L_test)
    @test norm(b - b_test) < 10e-13

    # solve
    x₀, stats = cg(L, b)

    # apply particular solution
    x = apply_particular_solution(model, x₀)
    x_test = apply_particular_solution(L_test, x₀)
    @test norm(x_test - x) < 10e-13
end

@testset "FastDiagonalization on Cartesian grid with constant parameters" begin
    # define mapping
    F = rectangle()
    F = refine(F, method=pRefinement(1))
    F = refine(F, method=hRefinement(2))

    # define dimension
    Dim = dimension(F)

    # get domain mapping is defined on
    Ω = domain(F)

    # define space constraints
    C = ScalarSplineSpaceConstraints{Dim}()
    clamped_constraint!(C, :left, :right, :top, :bottom)

    # define scalar spline space
    S = ScalarSplineSpace(F.space)

    # define solution field
    uʰ = Field(S)

    # define field to satisfy Dirichlet boundary conditions
    ūʰ = Field(S)
    ūʰ[1].coeffs .= 0

    # define Poisson parameters
    f = (x, y) -> 1.0 # bodyforce (-Δuʰ = f)
    κ = (x, y) -> SMatrix{2,2,Float64}([1.0 0.0; 0.0 2.0]) # conductivity
    t = (x, y) -> @SVector [0.0; 0.0] # traction

    # initialize model
    model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t)

    # construct linear operator
    L = linear_operator(model)

    # construct FastDiagonalization preconditioner using modal splines data approximation
    p̃, ñₑ = 1, 1
    P₁ = FastDiagonalization(model; method=ModalSplines, spaces=ntuple(k -> SplineSpace(p̃, domain(F.space[k]), ñₑ), 2))
    P₂ = FastDiagonalization(model; method=Wachspress, niter=2)

    # P === L⁻¹ thus P*L == I    
    @test norm(Matrix(P₁) * Matrix(L) - I) < 10e-13
    @test norm(Matrix(P₂) * Matrix(L) - I) < 10e-13
end

@testset "LinearOperatorApproximation using ModalSplines" begin
    # define mapping
    F = rectangle()
    F = refine(F, method=pRefinement(1))
    F = refine(F, method=hRefinement(2))

    # define dimension
    Dim = dimension(F)

    # get domain mapping is defined on
    Ω = domain(F)

    # define space constraints
    C = ScalarSplineSpaceConstraints{Dim}()
    clamped_constraint!(C, :left, :right, :top, :bottom)

    # define scalar spline space
    S = ScalarSplineSpace(F.space)

    # define solution field
    uʰ = Field(S)

    # define field to satisfy Dirichlet boundary conditions
    ūʰ = Field(S)
    ūʰ[1].coeffs .= 0

    # define Poisson parameters
    f = (x, y) -> 1.0 # bodyforce (-Δuʰ = f)
    κ = (x, y) -> SMatrix{2,2,Float64}([1.0 0.7; 0.3 1.3]) # conductivity
    t = (x, y) -> @SVector [0.0; 0.0] # traction

    # initialize model
    model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t)

    # construct linear operator
    L = linear_operator(model)

    # construct linear operator approximation
    p̃, ñₑ, rank = 1, 1, 1
    L̃ = linear_operator_approximation(model; method=ModalSplines, rank=rank, spaces=ntuple(k -> SplineSpace(p̃, domain(F.space[k]), ñₑ), 2))

    # since we work on Cartesian grids L̃₁ == L
    # cross terms are tested via non-diagonal conductivity
    @test norm(Matrix(L̃) - Matrix(L)) < 10e-13
end


@testset "LinearOperatorApproximation using Wachspress" begin
    # define mapping
    F = rectangle()
    F = refine(F, method=pRefinement(1))
    F = refine(F, method=hRefinement(2))

    # define dimension
    Dim = dimension(F)

    # get domain mapping is defined on
    Ω = domain(F)

    # define space constraints
    C = ScalarSplineSpaceConstraints{Dim}()
    clamped_constraint!(C, :left, :right, :top, :bottom)

    # define scalar spline space
    S = ScalarSplineSpace(F.space)

    # define solution field
    uʰ = Field(S)

    # define field to satisfy Dirichlet boundary conditions
    ūʰ = Field(S)
    ūʰ[1].coeffs .= 0

    # define Poisson parameters
    f = (x, y) -> 1.0 # bodyforce (-Δuʰ = f)
    κ = (x, y) -> SMatrix{2,2,Float64}(I) # conductivity
    t = (x, y) -> @SVector [0.0; 0.0] # traction

    # initialize model
    model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t)

    # construct linear operator
    L = linear_operator(model)

    # construct linear operator approximation
    p̃, ñₑ, rank = 1, 1, 1
    L̃ = linear_operator_approximation(model; method=Wachspress, niter=2)

    # since we work on Cartesian grids and have κ = I  and L̃ == L
    @test norm(Matrix(L̃) - Matrix(L)) < 10e-13
end

@testset "LinearOperatorApproximation using CanonicalPolyadic" begin
    # define mapping
    F = rectangle()
    F = refine(F, method=pRefinement(1))
    F = refine(F, method=hRefinement(10))

    # define dimension
    Dim = dimension(F)

    # get domain mapping is defined on
    Ω = domain(F)

    # define space constraints
    C = ScalarSplineSpaceConstraints{Dim}()
    clamped_constraint!(C, :left, :right, :top, :bottom)

    # define scalar spline space
    S = ScalarSplineSpace(F.space)

    # define solution field
    uʰ = Field(S)

    # define field to satisfy Dirichlet boundary conditions
    ūʰ = Field(S)
    ūʰ[1].coeffs .= 0

    # define Poisson parameters
    f = (x, y) -> 1.0 # bodyforce (-Δuʰ = f)
    κ = (x, y) -> SMatrix{2,2,Float64}(I) # conductivity
    t = (x, y) -> @SVector [0.0; 0.0] # traction

    # initialize model
    model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t)

    # construct linear operator
    L = linear_operator(model)

    # construct linear operator approximation
    L̃ = linear_operator_approximation(model; method=CanonicalPolyadic, tol=10e-5, rank=2, ntries=42)

    # since we work on Cartesian grids and have κ = I  and L̃ == L
    @test norm(Matrix(L̃) - Matrix(L)) < 10e-13
end