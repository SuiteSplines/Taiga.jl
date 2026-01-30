using LinearAlgebra

@testset "Test ExampleModel Taiga integration" begin
    Dim = 3
    T = Float64

    # test model initialization
    model = ExampleModel{Dim,T}()
    @test typeof(model) <: Model{Dim,T}

    # test construction of module linear operator
    A = linear_operator(model)
    @test typeof(A) <: LinearOperator{Dim,T}

    # test construction of module linear operator approximation
    Ã = linear_operator_approximation(model)
    @test typeof(Ã) <: LinearOperatorApproximation{Dim,T}

    # test size
    sz = size(A)
    @test sz == (42,42)

    v = rand(size(A, 2))
    b = zeros(size(A, 1))

    # test application of linear operator
    mul!(b, A, v)
    @test b == v

    # test application of linear operator approximation
    mul!(b, Ã, v)
    @test b == 0.5v

    # test forcing update
    forcing!(b, model)
    @test all(b .≈ π)

    # test forcing return
    b = forcing(model)
    @test all(b .≈ 2π)
end

@testset "Test LinearOperator pretty print" begin
    model = ExampleModel{3,Float64}()
    A = linear_operator(model)
    io = IOBuffer();
    Base.show(io, A)
    msg = String(take!(io))
    @test msg == "Linear operator of type Taiga.ExampleModule.LinearOperator{3, Float64} \
    with size (42, 42)"

    model = ExampleModel{3,Float64}()
    A = linear_operator_approximation(model)
    io = IOBuffer();
    Base.show(io, A)
    msg = String(take!(io))
    @test msg == "Linear operator approximation of type \
    Taiga.ExampleModule.LinearOperatorApproximation{3, Float64} with size (42, 42)"
end

@testset "Test Constrained flag" begin
    @test Constrained(true) isa Constrained{true}
    @test Constrained(false) isa Constrained{false}
    @test_throws MethodError Constrained(1) 
    @test_throws MethodError Constrained("yes")
end