using SortedSequences, CartesianProducts, KroneckerProducts, LinearAlgebra
import LinearMaps: LinearMap

@testset "ι tuple generation" begin
    # no derivatives
    u = ι(0, dim=2)
    @test u == (0, 0)

    u = ι(0, dim=3)
    @test u == (0, 0, 0)

    # gradient
    let
        ∇u(k) = ι(k, dim=2)
        @test ∇u(1) == (1, 0)
        @test ∇u(2) == (0, 1)
    end
    let
        ∇u(k) = ι(k, dim=3)
        @test ∇u(1) == (1, 0, 0)
        @test ∇u(2) == (0, 1, 0)
        @test ∇u(3) == (0, 0, 1)
    end

    # gradient but with specific value
    let
        ∇²ₖₖu(k) = ι(k; val=2, dim=2)
        @test ∇²ₖₖu(1) == (2, 0)
        @test ∇²ₖₖu(2) == (0, 2)
    end
    let
        ∇²ₖₖu(k) = ι(k, val=2, dim=3)
        @test ∇²ₖₖu(1) == (2, 0, 0)
        @test ∇²ₖₖu(2) == (0, 2, 0)
        @test ∇²ₖₖu(3) == (0, 0, 2)
    end
    
    # hessian
    let
        ∇²u(k, l) = ι(k, l, dim=2)
        @test ∇²u(1, 1) == (2, 0)
        @test ∇²u(2, 1) == (1, 1)
        @test ∇²u(1, 2) == (1, 1)
        @test ∇²u(2, 2) == (0, 2)
    end
    let
        ∇²u(k, l) = ι(k, l, dim=3)
        @test ∇²u(1, 1) == (2, 0, 0)
        @test ∇²u(1, 2) == (1, 1, 0)
        @test ∇²u(1, 3) == (1, 0, 1)
        @test ∇²u(2, 1) == (1, 1, 0)
        @test ∇²u(2, 2) == (0, 2, 0)
        @test ∇²u(2, 3) == (0, 1, 1)
        @test ∇²u(3, 1) == (1, 0, 1)
        @test ∇²u(3, 2) == (0, 1, 1)
        @test ∇²u(3, 3) == (0, 0, 2)
    end

    # hessian but with specific values
    let
        ∇²u(k, l) = ι(k, l, dim=2)
        @test ∇²u(1, 1) == (2, 0)
        @test ∇²u(2, 1) == (1, 1)
        @test ∇²u(1, 2) == (1, 1)
        @test ∇²u(2, 2) == (0, 2)
    end
    let
        ∇²u(k, l) = ι(k, l, val=(1, 2), dim=3)
        @test ∇²u(1, 1) == (3, 0, 0)
        @test ∇²u(1, 2) == (1, 2, 0)
        @test ∇²u(1, 3) == (1, 0, 2)
        @test ∇²u(2, 1) == (2, 1, 0)
        @test ∇²u(2, 2) == (0, 3, 0)
        @test ∇²u(2, 3) == (0, 1, 2)
        @test ∇²u(3, 1) == (2, 0, 1)
        @test ∇²u(3, 2) == (0, 2, 1)
        @test ∇²u(3, 3) == (0, 0, 3)
    end
end

@testset "Kroncker product operator norm" begin
    A = reshape(collect(1:5*7), 5, 7)
    B = reshape(collect(1:11*3), 11, 3)

    # using specialized opnorm
    val = opnorm(A ⊗ B)

    # using Julia opnorm
    val_test = opnorm(Matrix(A ⊗ B))

    # test    
    @test val ≈ val_test
end

@testset "Compute extreme eigenvalues of a linear map" begin
    D = LinearMap(Diagonal([2 + im * 1, 3, 5]))
    λmax, λmin = extreme_eigenvalues(D)
    @test λmax ≈ 5.0 + im * 0.0
    @test λmin ≈ 2.0 + im * 1.0

    D = LinearMap(Diagonal([2 + im * eps(Float64), 3, 5]))
    λmax, λmin = extreme_eigenvalues(D)
    @test λmax ≈ 5.0
    @test λmin ≈ 2.0
end