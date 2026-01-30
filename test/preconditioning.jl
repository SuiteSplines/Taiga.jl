using LinearAlgebra, KroneckerProducts, SortedSequences, UnivariateSplines, CartesianProducts
using IgaBase, AbstractMappings
using StaticArrays, Krylov

@testset "Preconditioners general interface" begin
    struct ExamplePreconditioner{Dim,T} <: Preconditioner{Dim,T} end
    P = ExamplePreconditioner{3,Float64}()

    # test type
    @test typeof(P) <: Preconditioner

    # test missing Base.size imp
    e = "ExamplePreconditioner{3, Float64} does not implement Base.size()!"
    @test_throws ErrorException(e) size(P)

    # test missing LinearAlgebra.issymmetric imp
    e = "ExamplePreconditioner{3, Float64} does not implement LinearAlgebra.issymmetric()!"
    @test_throws ErrorException(e) issymmetric(P)

    # test missing LinearAlgebra.ishermitian imp
    e = "ExamplePreconditioner{3, Float64} does not implement LinearAlgebra.ishermitian()!"
    @test_throws ErrorException(e) ishermitian(P)

    # test missing LinearAlgebra.isposdef imp
    e = "ExamplePreconditioner{3, Float64} does not implement LinearAlgebra.isposdef()!"
    @test_throws ErrorException(e) isposdef(P)

    Base.size(::ExamplePreconditioner) = (42, 42)

    # test pretty print
    io = IOBuffer()
    Base.show(io, P)
    msg = String(take!(io))
    @test msg == "Preconditioner of type ExamplePreconditioner{3, Float64} with size (42, 42)"
end

@testset "Generic FastDiagonalization preconditioner" begin
    # system matrix
    M, M₁, M₂, K₁, K₂ = cartesian_laplace()
    M⁻¹ = inv(M)

    # eigenvalue decompositions
    EVD₁ = eigen(K₁, M₁)
    EVD₂ = eigen(K₂, M₂)

    # eigenvalues
    Λ₁ = Diagonal(EVD₁.values)
    Λ₂ = Diagonal(EVD₂.values)
    Λ = Diagonal(Λ₂ ⊕ Λ₁)
    Λ⁻¹ = Diagonal(inv.(Λ.diag))

    # eigenvectors
    U₁ = EVD₁.vectors
    U₂ = EVD₂.vectors
    U = U₂ ⊗ U₁

    # identity matrices
    I₁ = Diagonal(ones(size(U₁,1)))
    I₂ = Diagonal(ones(size(U₂,1)))

    # test Mₖ-orthogonality decomposition (quirky Julia)
    @test U₁' * M₁ * U₁ ≈ U₁ * U₁' * M₁ ≈ I
    @test U₂' * M₂ * U₂ ≈ U₂ * U₂' * M₂ ≈ I
    @test M₁ ≈ inv(U₁') * inv(U₁)
    @test M₂ ≈ inv(U₂') * inv(U₂)

    # test decompositions Kₖ = Mₖ Uₖ Λₖ Uₖ⁻¹
    @test K₁ ≈ M₁ * U₁ * Λ₁ * inv(U₁)
    @test K₂ ≈ M₂ * U₂ * Λ₂ * inv(U₂)

    # test decomposition of M in terms of Kronecker eigenvalue decompositions
    @test M ≈ (inv(U₂)' ⊗ inv(U₁)') * ( I₂ ⊗ Λ₁ + Λ₂ ⊗ I₁) * (inv(U₂) ⊗ inv(U₁))

    # test Kronecker eigendecomposition (i.e. KroneckerSum)
    @test M ≈ inv(U)' * Λ * inv(U)

    # test Kronecker eigendecomposition inversion (i.e. KroneckerSum)
    @test M⁻¹ ≈ U * Λ⁻¹ * U'

    # define FastDiagonalization preconditioner
    P = FastDiagonalization(Λ⁻¹, U)

    # test FastDiagonalization preconditioner
    @test Matrix(P) ≈ M⁻¹
    @test Matrix(P) * M ≈ I
    @test isposdef(P) == true
    @test ishermitian(P) == true
    @test issymmetric(P) == true
end

@testset "InnerCG and InnerPCG" begin
    # benchmark model
    model = poisson_annulus()

    # construct linear operator
    L = linear_operator(model)

    # construct rhs vector
    b = forcing(L)

    # construct FastDiagonalization preconditioner using modal splines data approximation
    p̃, ñₑ, rank = 3, 15, 3
    P₀ = FastDiagonalization(model; method=ModalSplines, spaces=ntuple(k -> SplineSpace(p̃, domain(model.F.space[k]), ñₑ), 2))
    L̃ = linear_operator_approximation(model; method=ModalSplines, rank=rank, spaces=ntuple(k -> SplineSpace(p̃, domain(model.F.space[k]), ñₑ), 2))
    P₁ = InnerCG(L̃; η=10e-5, history=true, eigtol=10e-8)
    P₂ = InnerPCG(L̃, P₀; η=10e-5, history=true, eigtol=10e-8)

    # check properties
    @test ishermitian(P₁) == true
    @test issymmetric(P₁) == true
    @test isposdef(P₁) == true
    @test ishermitian(P₂) == true
    @test issymmetric(P₂) == true
    @test isposdef(P₂) == true

    # solve without preconditioner
    solver = TaigaCG(L; atol=10e-3, rtol=10e-3)
    x₀, stats = linsolve!(solver, b)
    niter_cg = stats.niter

    # solve with fast diagonalization preconditioner
    solver = TaigaPCG(L, P₀; atol=10e-3, rtol=10e-3)
    x₀, stats = linsolve!(solver, b)
    niter_pcg = stats.niter

    # solve with inexact Kronecker preconditioner
    solver = TaigaIPCG(L, P₁; atol=10e-3, rtol=10e-3)
    x₀, stats = linsolve!(solver, b)
    niter_ipcg_cg = stats.niter

    solver = TaigaIPCG(L, P₂; atol=10e-3, rtol=10e-3)
    x₀, stats = linsolve!(solver, b)
    niter_ipcg_pcg = stats.niter

    # crude test
    @test niter_cg > niter_pcg > niter_ipcg_cg
    @test niter_ipcg_pcg ≤ niter_ipcg_cg

    # test convergence criterion for InnerCG
    r, η, η̂ = b, P₁.η, P₁.η̂
    x = P₁ * r
    @test norm(L̃ * x - b) ≤ η̂ * norm(b)

    # test convergence criterion for InnerPCG
    r, η, η̂ = b, P₂.η, P₂.η̂
    λmax, λmin = extreme_eigenvalues(P₀ * L̃, tol=10e-8)
    @assert λmax > λmin
    γ = λmax / λmin
    x = P₂ * r
    @test (η̂ - η / sqrt(γ)) < 10e-7
    @test norm(P₀ * L̃ * x - P₀ * b) ≤ η̂ * norm(P₀ * b)

    # test convergence in 0 iterations (dummy data...)
    reset_inner_solver_history!.((P₁, P₂))
    b = zeros(size(P₁, 2))
    x = P₁ * b
    x = P₂ * b
    @test inner_solver_niters(P₁)[1] == 0
    @test inner_solver_niters(P₂)[1] == 0
end

@testset "InnerCG and InnerPCG history" begin
    # benchmark model
    model = poisson_annulus()

    # construct linear operator
    L = linear_operator(model)

    # construct rhs vector
    b = forcing(L)

    # construct FastDiagonalization preconditioner using modal splines data approximation
    p̃, ñₑ, rank = 3, 15, 3
    P₀ = FastDiagonalization(model; method=ModalSplines, spaces=ntuple(k -> SplineSpace(p̃, domain(model.F.space[k]), ñₑ), 2))
    L̃ = linear_operator_approximation(model; method=ModalSplines, rank=rank, spaces=ntuple(k -> SplineSpace(p̃, domain(model.F.space[k]), ñₑ), 2))
    P₁ = InnerCG(L̃; η=10e-5, history=true)
    P₂ = InnerPCG(L̃, P₀; η=10e-5, history=true)

    solver = TaigaIPCG(L, P₁; atol=10e-3, rtol=10e-3)
    x₀, stats = linsolve!(solver, b)

    solver = TaigaIPCG(L, P₂; atol=10e-3, rtol=10e-3)
    x₀, stats = linsolve!(solver, b)

    reset_inner_solver_history!(P₁)
    reset_inner_solver_history!(P₂)

    @test inner_solver_niters(P₁) == []
    @test inner_solver_residuals(P₁) == []
    @test inner_solver_convergence(P₁) == []
    @test inner_solver_niters(P₂) == []
    @test inner_solver_residuals(P₂) == []
    @test inner_solver_convergence(P₂) == []

    solver = TaigaIPCG(L, P₁; atol=10e-3, rtol=10e-3)
    x₀, stats_P₁ = linsolve!(solver, b)

    solver = TaigaIPCG(L, P₂; atol=10e-3, rtol=10e-3)
    x₀, stats_P₂ = linsolve!(solver, b)

    m = minimum((stats_P₂.niter, stats_P₂.niter))
    @test all(inner_solver_niters(P₂)[1:m] .< inner_solver_niters(P₁)[1:m])

    r = inner_solver_residuals(P₁)
    c = inner_solver_convergence(P₁)
    for k in 2:length(r)
        @test r[k-1] > r[k] # stopping crit depends on outer residual norm!
        @test c[k-1] == c[k] == true
    end

    r = inner_solver_residuals(P₂)
    c = inner_solver_convergence(P₂)
    for k in 2:length(r)
        @test r[k-1] > r[k] # stopping crit depends on outer residual norm!
        @test c[k-1] == c[k] == true
    end
end

@testset "HyperPowerPreconditioner" begin
    # benchmark model
    model = poisson_annulus()

    # construct linear operator
    L = linear_operator(model)

    # construct rhs vector
    b = forcing(L)

    # construct FastDiagonalization preconditioner using modal splines data approximation
    p̃, ñₑ, rank = 3, 15, 1
    P₀ = FastDiagonalization(model; method=ModalSplines, spaces=ntuple(k -> SplineSpace(p̃, domain(model.F.space[k]), ñₑ), 2))
    L̃ = linear_operator_approximation(model; method=ModalSplines, rank=rank, spaces=ntuple(k -> SplineSpace(p̃, domain(model.F.space[k]), ñₑ), 2))
    P = HyperPowerPreconditioner(L̃, P₀, 4)

    # check properties
    @test ishermitian(P) == true
    @test issymmetric(P) == true
    @test isposdef(P) == true
    tol = 10e-5
    λmin, λmax = hyperpower_extreme_eigenvalues(P; tol=tol, n=0)
    @test λmin > 0
    @test λmax < 2

    # check extreme eigenvalue computation
    λ₁ = hyperpower_extreme_eigenvalues(P; tol=10e-5, n=1)
    λ₂ = hyperpower_extreme_eigenvalues(P; tol=10e-5, n=2)
    λ₃ = hyperpower_extreme_eigenvalues(P; tol=10e-5, n=3)
    λ₄ = hyperpower_extreme_eigenvalues(P; tol=10e-5, n=4)
    @test norm(λ₁ .- hyperpower_eigenvalues_after_n_iterations(λ₁; n=0)) < tol
    @test norm(λ₂ .- hyperpower_eigenvalues_after_n_iterations(λ₁; n=1)) < tol
    @test norm(λ₃ .- hyperpower_eigenvalues_after_n_iterations(λ₁; n=2)) < tol
    @test norm(λ₄ .- hyperpower_eigenvalues_after_n_iterations(λ₁; n=3)) < tol

    # check initial preconditioner
    @test hyperpower_initial_preconditioner(P) == P₀

    # solve
    x₀, log = cg(L, b; M=P₀)
    niter_test = log.niter

    # solve with preconditioner
    x₀, log = cg(L, b; M=P)
    @test log.niter < niter_test # a very crude test
    @test log.niter < 9

    # test Base.show (for n > 1, it includes the number of updates)
    P = HyperPowerPreconditioner(L̃, P₀)
    io = IOBuffer();
    Base.show(io, P)
    msg = String(take!(io))
    @test msg == "Preconditioner of type HyperPowerPreconditioner{2, \
    Float64, Taiga.PoissonModule.LinearOperatorApproximation{2, Float64}, \
    FastDiagonalization{2, Float64, KroneckerProduct{Float64, \
    2, 2, Tuple{Matrix{Float64}, Matrix{Float64}}}}} with size (935, 935)"

    P = HyperPowerPreconditioner(L̃, P₀, 2)
    io = IOBuffer();
    Base.show(io, P)
    msg = String(take!(io))
    @test msg == "Preconditioner of type HyperPowerPreconditioner{2,Float64,\
    Taiga.PoissonModule.LinearOperatorApproximation{2, Float64},\
    B<:HyperPowerPreconditioner{Dim,T}} and size (935, 935).\nThe \
    initial preconditioner FastDiagonalization{2, Float64, \
    KroneckerProduct{Float64, 2, 2, Tuple{Matrix{Float64}, \
    Matrix{Float64}}}} was updated 2 times."
end
