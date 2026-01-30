using IgaBase, CartesianProducts, SortedSequences, UnivariateSplines, KroneckerProducts, SpecialSpaces
using LinearAlgebra

constraints = ScalarSplineSpaceConstraints{2}()
left_constraint!(constraints; dim=1)
right_constraint!(constraints; dim=1)
left_constraint!(constraints; dim=2)
right_constraint!(constraints; dim=2)

Ω = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)
Δ = Partition(Ω, (7, 19))
p = (2, 2)

S = ScalarSplineSpace(p, Δ, constraints)
V = ScalarSplineSpace(p .+ 2, Δ, constraints)

Dim = 2
m = dimension(V)
n = dimension(S)

@testset "KroneckerFactory construction" begin
    ∫ = KroneckerFactory(S, V)
    @test size(∫.data) == (m, n)
end

@testset "KroneckerFactory reset" begin
    ∫ = KroneckerFactory(S, V)
    testadd! = (n; reset) -> map(k -> ∫((0,0),(0,0); reset=reset), 1:n)

    testadd!(3, reset=false)
    @test length(∫.data.K) == 3

    testadd!(3, reset=true)
    @test length(∫.data.K) == 1

    reset!(∫)
    @test length(∫.data.K) == 0
end

@testset "KroneckerFactory without weighting in two dimensions" begin
    C = reverse(extraction_operator(S).data)
    C₁, C₂ = C[1], C[2]
    V₁, V₂ = S[1], S[2]

    # compute system matrices without weighting
    M₂ = Matrix(C₂' * UnivariateSplines.system_matrix(V₂, V₂, 1, 1) * C₂)
    K₁ = Matrix(C₁' * UnivariateSplines.system_matrix(V₁, V₁, 2, 2) * C₁)

    B₂ = Matrix(C₂' * UnivariateSplines.system_matrix(V₂, V₂, 1, 2) * C₂)
    A₁ = Matrix(C₁' * UnivariateSplines.system_matrix(V₁, V₁, 2, 1) * C₁)

    A₂ = Matrix(C₂' * UnivariateSplines.system_matrix(V₂, V₂, 2, 1) * C₂)
    B₁ = Matrix(C₁' * UnivariateSplines.system_matrix(V₁, V₁, 1, 2) * C₁)

    K₂ = Matrix(C₂' * UnivariateSplines.system_matrix(V₂, V₂, 2, 2) * C₂)
    M₁ = Matrix(C₁' * UnivariateSplines.system_matrix(V₁, V₁, 1, 1) * C₁)

    # system matrix
    M = M₂ ⊗ K₁ + B₂ ⊗ A₁ + A₂ ⊗ B₁ + K₂ ⊗ M₁

    # a test without weighting
    ∇u = k -> ι(k, dim=Dim)
    ∇v = k -> ι(k, dim=Dim)

    ∫ = KroneckerFactory(S, S)
    for β in 1:Dim
        for α in 1:Dim
            ∫(∇u(α), ∇v(β))
        end
    end

    @test Matrix(∫.data) ≈ M
end

@testset "KroneckerFactory with spline weighting" begin
    C = reverse(extraction_operator(S).data)
    C₁, C₂ = C[1], C[2]
    V₁, V₂ = S[1], S[2]

    # define weight function spaces
    S̃ = ScalarSplineSpace((1,1), Δ)

    # define univariate weight functions
    f₁ = [Bspline(S̃[1]) for k = 1:Dim, k = 1:Dim]
    f₂ = [Bspline(S̃[2]) for k = 1:Dim, k = 1:Dim]

    # set some coefficients
    w₁ = [2.0 0.25; 0.50 3.0]
    w₂ = [3.0 0.75; 0.25 4.0]
    w = w₁ .* w₂ # actual weights per Kronecker product

    # set weight function coefficients
    for l in 1:Dim
        for k in 1:Dim
            f₁[k, l].coeffs .= w₁[k, l]
            f₂[k, l].coeffs .= w₂[k, l]
        end
    end

    # collect weight function per data array element in a tuple
    f = [(f₁[k,l], f₂[k,l]) for k = 1:Dim, l = 1:Dim]

    # compute weighted system matrices
    M₂ = Matrix(C₂' * UnivariateSplines.system_matrix(V₂, V₂, 1, 1) * C₂)
    K₁ = Matrix(C₁' * UnivariateSplines.system_matrix(V₁, V₁, 2, 2) * C₁)

    B₂ = Matrix(C₂' * UnivariateSplines.system_matrix(V₂, V₂, 1, 2) * C₂)
    A₁ = Matrix(C₁' * UnivariateSplines.system_matrix(V₁, V₁, 2, 1) * C₁)

    A₂ = Matrix(C₂' * UnivariateSplines.system_matrix(V₂, V₂, 2, 1) * C₂)
    B₁ = Matrix(C₁' * UnivariateSplines.system_matrix(V₁, V₁, 1, 2) * C₁)

    K₂ = Matrix(C₂' * UnivariateSplines.system_matrix(V₂, V₂, 2, 2) * C₂)
    M₁ = Matrix(C₁' * UnivariateSplines.system_matrix(V₁, V₁, 1, 1) * C₁)

    # system matrix weighted by spline functions
    M = (w[1,1] * M₂ ⊗ K₁) + (w[2,1] * B₂ ⊗ A₁) + (w[1,2] * A₂ ⊗ B₁) + (w[2,2] * K₂ ⊗ M₁)

    # a test with weighting
    ∇u = k -> ι(k, dim=Dim)
    ∇v = k -> ι(k, dim=Dim)

    ∫ = KroneckerFactory(S, S)
    for β in 1:Dim
        for α in 1:Dim
            ∫(∇u(α), ∇v(β); data=f[α, β])
        end
    end

    @test norm(Matrix(∫.data) - M) < 10e-12
end

@testset "KroneckerFactory with vector weighting" begin
    C = reverse(extraction_operator(S).data)
    C₁, C₂ = C[1], C[2]
    V₁, V₂ = S[1], S[2]

    ## set some coefficients
    w₁ = [2.0 0.25; 0.50 3.0]
    w₂ = [3.0 0.75; 0.25 4.0]
    w = w₁ .* w₂ # actual weights per Kronecker product

    # define weight vectors
    q₁ = Taiga.get_system_matrix_quadrule(S[1], S[1])
    q₂ = Taiga.get_system_matrix_quadrule(S[2], S[2])
    q₁_num_wts = length(q₁.w)
    q₂_num_wts = length(q₂.w)

    # define univariate weight vectors
    v₁ = [ones(q₁_num_wts) for k = 1:Dim, k = 1:Dim]
    v₂ = [ones(q₂_num_wts) for k = 1:Dim, k = 1:Dim]

    # set weight vectors to coefficients
    for l in 1:Dim
        for k in 1:Dim
            v₁[k, l] .= w₁[k, l]
            v₂[k, l] .= w₂[k, l]
        end
    end

    # collect weight vectors per data array element in a tuple
    v = [(v₁[k,l], v₂[k,l]) for k = 1:Dim, l = 1:Dim]

    # compute weighted system matrices
    M₂ = Matrix(C₂' * UnivariateSplines.system_matrix(V₂, V₂, 1, 1) * C₂)
    K₁ = Matrix(C₁' * UnivariateSplines.system_matrix(V₁, V₁, 2, 2) * C₁)

    B₂ = Matrix(C₂' * UnivariateSplines.system_matrix(V₂, V₂, 1, 2) * C₂)
    A₁ = Matrix(C₁' * UnivariateSplines.system_matrix(V₁, V₁, 2, 1) * C₁)

    A₂ = Matrix(C₂' * UnivariateSplines.system_matrix(V₂, V₂, 2, 1) * C₂)
    B₁ = Matrix(C₁' * UnivariateSplines.system_matrix(V₁, V₁, 1, 2) * C₁)

    K₂ = Matrix(C₂' * UnivariateSplines.system_matrix(V₂, V₂, 2, 2) * C₂)
    M₁ = Matrix(C₁' * UnivariateSplines.system_matrix(V₁, V₁, 1, 1) * C₁)

    # system matrix weighted by spline functions
    M = (w[1,1] * M₂ ⊗ K₁) + (w[2,1] * B₂ ⊗ A₁) + (w[1,2] * A₂ ⊗ B₁) + (w[2,2] * K₂ ⊗ M₁)

    # a test with weighting
    ∇u = k -> ι(k, dim=Dim)
    ∇v = k -> ι(k, dim=Dim)

    ∫ = KroneckerFactory(S, S)
    for β in 1:Dim
        for α in 1:Dim
            ∫(∇u(α), ∇v(β); data=v[α, β])
        end
    end

    @test norm(Matrix(∫.data) - M) < 10e-12
end