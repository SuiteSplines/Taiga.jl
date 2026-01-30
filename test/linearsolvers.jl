@testset "LinearSolver stats pretty show" begin
    model = poisson_annulus()
    L = linear_operator(model)
    b = forcing(L)
    solver = TaigaCG(L; atol=10e-3, rtol=10e-2, itmax=10)

    # test show
    io = IOBuffer()
    Base.show(io, solver)
    msg = String(take!(io))
    @test msg == "Linear solver of type TaigaCG (atol=0.01, rtol=0.1, itmax=10)"
end

@testset "LinearSolverStatistics" begin
    stats = LinearSolverStatistics(TaigaCG)
    props = propertynames(stats)
    @test typeof(stats) == LinearSolverStatistics{TaigaCG}
    @test :converged in props
    @test :niter in props
    @test :residual in props
    @test :status in props
    @test stats.converged == false
    @test stats.niter == 0
    @test stats.residual == 0.0
    @test stats.status == ""

    setproperty!(stats, :converged, true)
    setproperty!(stats, :niter, 13)
    setproperty!(stats, :residual, 42.314)
    setproperty!(stats, :status, "🐛")
    getproperty(stats, :converged) == true
    getproperty(stats, :niter) == 13
    getproperty(stats, :residual) == 42.314
    getproperty(stats, :status) == "🐛"

    stats.newprop = 3//5
    @test stats.newprop == 3//5
end

@testset "TaigaCG stats pretty show" begin
    model = poisson_annulus()
    L = linear_operator(model)
    b = forcing(L)
    solver = TaigaCG(L; atol=10e-8, rtol=10e-8, itmax=10)
    x, stats = linsolve!(solver, b)

    # test contents
    io = IOBuffer()
    Base.show(io, stats)
    msg = String(take!(io))
    @test msgcontains = begin
        occursin("Metric", msg) &&
        occursin("Value", msg) &&
        occursin("10", msg) &&
        occursin("converged", msg) &&
        occursin("not converged", msg) &&
        occursin("niter", msg) &&
        occursin("residual", msg) &&
        occursin("status", msg) &&
        occursin("niter == itmax", msg)
    end
end

@testset "TaigaCG" begin
    model = poisson_annulus()
    L = linear_operator(model)
    b = forcing(L)

    atol = 10e-8
    rtol = 0.0
    itmax = 300
    ϵ = atol + rtol * norm(b)

    solver = TaigaCG(L; atol=atol, rtol=rtol, itmax=itmax)
    @test typeof(solver) <: LinearSolver
    @test solver.atol == atol
    @test solver.rtol == rtol
    @test solver.itmax == itmax

    # empty statistics
    @test solver.stats.converged == false
    @test solver.stats.niter == 0
    @test solver.stats.residual == 0.0
    @test solver.stats.status == ""

    # test solve
    x, stats = linsolve!(solver, b)
    @test stats.converged == true
    @test abs(stats.niter - 202) <= 1
    @test abs(stats.residual - norm(L*x - b)) < 10e-13
    @test stats.residual ≤ ϵ
    @test stats.status == "converged: niter < itmax"

    # test solve with x0 good enough
    x, stats = linsolve!(solver, b; x0=x)
    @test stats.converged == true
    @test stats.niter == 0
    @test stats.residual ≤ ϵ
    @test stats.status == "converged: niter < itmax"

    # test solve with rtol
    atol = 10e-8
    rtol = 10e-4
    itmax = 100
    ϵ = atol + rtol * norm(b)

    solver = TaigaCG(L; atol=atol, rtol=rtol, itmax=itmax)
    x, stats = linsolve!(solver, b)
    @test stats.converged == true
    @test abs(stats.niter - 83) <= 1
    @test stats.residual ≤ ϵ
    @test stats.status == "converged: niter < itmax"

    # test solve with itmax too low
    atol = 10e-8
    rtol = 10e-4
    itmax = 10
    ϵ = atol + rtol * norm(b)

    solver = TaigaCG(L; atol=atol, rtol=rtol, itmax=itmax)
    x, stats = linsolve!(solver, b)
    @test stats.converged == false
    @test stats.niter == itmax
    @test stats.residual > ϵ
    @test stats.status == "not converged: niter == itmax"
end

@testset "TaigaPCG" begin
    model = poisson_annulus()
    p̃, nₑ = 2, 10
    P = FastDiagonalization(model; method=Wachspress, niter=2)
    L = linear_operator(model)
    b = forcing(L)

    atol = 10e-8
    rtol = 0.0
    itmax = 300
    ϵ = atol + rtol * norm(b)

    solver = TaigaCG(L; atol=atol, rtol=rtol, itmax=itmax)
    x, stats = linsolve!(solver, b)
    niter_test = stats.niter
    
    # test solve using Wachspress fast diagonalization preconditioner
    solver = TaigaPCG(L, P; atol=atol, rtol=rtol, itmax=itmax)
    x, stats = linsolve!(solver, b)
    @test stats.converged == true
    @test abs(stats.niter - 35) <= 1
    @test abs(stats.residual - norm(L*x - b)) < 10e-13
    @test stats.residual ≤ ϵ
    @test stats.status == "converged: niter < itmax"
    @test stats.niter < niter_test

    # test solve with x0 good enough
    x, stats = linsolve!(solver, b; x0=x)
    @test stats.converged == true
    @test stats.niter == 0
    @test stats.residual ≤ ϵ
    @test stats.status == "converged: niter < itmax"

    # test solve with rtol
    atol = 10e-8
    rtol = 10e-4
    itmax = 100
    ϵ = atol + rtol * norm(b)

    solver = TaigaPCG(L, P; atol=atol, rtol=rtol, itmax=itmax)
    x, stats = linsolve!(solver, b)
    @test stats.converged == true
    @test abs(stats.niter - 13) <= 1
    @test stats.residual ≤ ϵ
    @test stats.status == "converged: niter < itmax"

    # test solve with itmax too low
    atol = 10e-8
    rtol = 10e-4
    itmax = 5
    ϵ = atol + rtol * norm(b)

    solver = TaigaCG(L; atol=atol, rtol=rtol, itmax=itmax)
    x, stats = linsolve!(solver, b)
    @test stats.converged == false
    @test stats.niter == itmax
    @test stats.residual > ϵ
    @test stats.status == "not converged: niter == itmax"
end

@testset "TaigaIPCG" begin
    model = poisson_annulus()
    p̃, ñₑ, rank = 2, 100, 3
    spaces = ntuple(k -> SplineSpace(p̃, domain(model.F.space[k]), ñₑ), 2)

    L = linear_operator(model)
    b = forcing(L)

    L̃ = linear_operator_approximation(model; method=ModalSplines, spaces=spaces, rank=rank)
    P₀ = FastDiagonalization(model; method=Wachspress, niter=2)
    P₁ = InnerCG(L̃; history=true)
    P₂ = InnerPCG(L̃, P₀; history=true)

    atol = 10e-8
    rtol = 0.0
    itmax = 300
    ϵ = atol + rtol * norm(b)

    solver = TaigaPCG(L, P₀; atol=atol, rtol=rtol, itmax=itmax)
    x, stats = linsolve!(solver, b)
    niter_test = stats.niter

    solver = TaigaIPCG(L, P₁; atol=atol, rtol=rtol, itmax=itmax)
    x, stats = linsolve!(solver, b)
    niter_inner_cg = stats.niter
    @test niter_test > niter_inner_cg

    solver = TaigaIPCG(L, P₂; atol=atol, rtol=rtol, itmax=itmax)
    x, stats = linsolve!(solver, b)
    niter_inner_pcg = stats.niter
    @test niter_test > niter_inner_pcg
    
    @test abs(niter_inner_cg - niter_inner_pcg) ≤ 1
    m = minimum((niter_inner_cg, niter_inner_pcg))
    @test all(inner_solver_niters(P₂)[1:m] .< inner_solver_niters(P₁)[1:m])

    @test stats.converged == true
    @test abs(stats.niter - 3) < 1
    @test abs(stats.residual - norm(L*x - b)) < 10e-13
    @test stats.residual ≤ ϵ
    @test stats.status == "converged: niter < itmax"
    @test stats.niter < niter_test
    
    # test solve with x0 good enough
    x, stats = linsolve!(solver, b; x0=x)
    @test stats.converged == true
    @test stats.niter == 0
    @test stats.residual ≤ ϵ
    @test stats.status == "converged: niter < itmax"

    # test solve with rtol
    atol = 10e-8
    rtol = 10e-4
    itmax = 100
    ϵ = atol + rtol * norm(b)

    solver = TaigaIPCG(L, P₂; atol=atol, rtol=rtol, itmax=itmax)
    x, stats = linsolve!(solver, b)
    @test stats.converged == true
    @test abs(stats.niter - 1) < 1
    @test stats.residual ≤ ϵ
    @test stats.status == "converged: niter < itmax"

    # test solve with itmax too low
    atol = 10e-8
    rtol = 0.0
    itmax = 1
    ϵ = atol + rtol * norm(b)

    solver = TaigaIPCG(L, P₂; atol=atol, rtol=rtol, itmax=itmax)
    x, stats = linsolve!(solver, b)
    @test stats.converged == false
    @test stats.niter == itmax
    @test stats.residual > ϵ
    @test stats.status == "not converged: niter == itmax"
end