using LinearAlgebra, KroneckerProducts

A = rand(3, 5)
B = rand(7, 3)

C = rand(3, 5)
D = rand(7, 3)

E = rand(3, 5)
F = rand(7, 4)

K₁ = A ⊗ B
K₂ = C ⊗ B
K₃ = E ⊗ F
K_test = K₁ + K₂

@testset "KroneckerProductAggregate construction and properties" begin
    T = eltype(K₁)
    sz = size(K₁)

    # initialize empty KroneckerProductAggregate
    K = KroneckerProductAggregate{T}(sz...)

    # test if empty
    @test isempty(K) == true

    # add K₁ and K₂ to aggregate
    push!(K, K₁)
    push!(K, K₂)

    # check factors
    @test get_factor(K, 1, 1) == A
    @test get_factor(K, 2, 2) == B

    # test that K is not empty
    @test isempty(K) == false

    # test assertion of conforming size
    msg = "Kronecker product to be aggregated is not of correct product size"
    @test_throws AssertionError(msg) push!(K, K₃)

    # test assertion of conforming size in constructor
    msg = "KroneckerProducts to be aggregated are not of the same product size"
    @test_throws AssertionError(msg) KroneckerProductAggregate(K₁, K₂, K₃)

    # test size of aggregate
    @test size(K) == sz

    # test element type of aggregate
    @test eltype(K) == T

    # test mvp
    @test Matrix(K) == K_test
    @test Matrix(KroneckerProductAggregate(K₁, K₂)) == K_test

    # test properties
    @test isposdef(KroneckerProductAggregate(K₁, K₂)) == false
    @test ishermitian(KroneckerProductAggregate(K₁, K₂)) == false
    @test issymmetric(KroneckerProductAggregate(K₁, K₂)) == false
    @test isposdef(KroneckerProductAggregate(K₁, K₂; isposdef=true)) == true
    @test ishermitian(KroneckerProductAggregate(K₁, K₂; ishermitian=true)) == true
    @test issymmetric(KroneckerProductAggregate(K₁, K₂; issymmetric=true)) == true
end

@testset "KroneckerProductAggregate matrix-vector products" begin
    K = KroneckerProductAggregate(K₁, K₂)
    b = zeros(size(K, 1))
    v = rand(size(K, 2))

    b_test = K_test * v

    # test matrix-vector product
    @test K * v ≈ b_test

    # test matrix-vector product into cache
    mul!(b, K, v)
    @test b ≈ b_test
end

@testset "KroneckerProductAggregate adjoint matrix-vector products" begin
    K = KroneckerProductAggregate(K₁, K₂)
    b = rand(size(K, 1))
    v = zeros(size(K, 2))

    v_test = K_test' * b

    # test matrix-vector product
    @test K' * b ≈ v_test

    # test matrix-vector product into cache
    mul!(v, K', b)
    @test v ≈ v_test
end


@testset "KroneckerProductAggregate droptol! based on operator norm" begin
    D₁ = Matrix(Diagonal(collect(1.5:-0.25:0.5)))
    D₂ = Matrix(Diagonal(collect(2.0:-0.5:0.5)))

    M = D₂ ⊗ D₁ # ‖M‖₂ = 3
    N = 10e-3 * D₂ ⊗ D₁ # ‖N‖₂ = 3e-2
    O = 10e-8 * D₂ ⊗ D₁ # ‖O‖₂ = 3e-7 

    K = KroneckerProductAggregate(M, N, O)
    n = length(K)

    # drop O
    droptol!(K; rtol=1e-6)
    @test length(K) == (n - 1)
    @test norm(Matrix(K) - (M + N)) < 10e-12

    # drop N
    droptol!(K; rtol=1e-1)
    @test length(K) == (n - 2)
    @test norm(Matrix(K) - M) < 10e-12
end