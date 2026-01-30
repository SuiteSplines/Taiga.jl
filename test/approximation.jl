using CartesianProducts, SortedSequences, UnivariateSplines, KroneckerProducts
using LinearAlgebra, NURBS, IgaFormation, SpecialSpaces, IgaBase, Statistics


@testset "Rank 1 modal splines approximation in two dimensions" begin
    # some analytical mapping with cross terms
    Ω = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)
    F = GeometricMapping(Ω, (x, y) -> x + y, (x, y) -> y)

    # spline spaces
    Δ = Partition(Ω, (5, 6))
    p = (3, 4)
    S = ScalarSplineSpace(p, Δ)
    S̃ = ntuple(k -> S[k], 2)

    # generate some data
    X = CartesianProduct(grevillepoints.(S̃)...)
    @evaluate C = Gradient(F)(X)
    for k in eachindex(X)
        J = C[k]
        C[k] = inv(J)' * inv(J) * det(J)
    end

    # compute modal splines approximation
    C̃ = approximate_patch_data(C; method=ModalSplines, spaces=S̃, rank=1)

    for l in 1:2
        for k in 1:2
            # evaluate modal weighting splines at grevillepoints
            @evaluate U = C̃[1, k, l][1](X.data[1])
            @evaluate V = C̃[1, k, l][2](X.data[2])

            # collect exact block data at grevillepoints
            c = C.data[k, l]

            # compute outer product of weighting splines at grevillepoints
            c̃ = U * V'

            # test matrix error norm
            @test norm(c - c̃) < 10e-13
        end
    end
end

@testset "Rank > 1 modal splines approximation in two dimensions" begin
    # some analytical mapping with cross terms
    Ω = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)
    f = ScalarFunction((x, y) -> x^2 + y^2)

    # spline spaces
    Δ = Partition(Ω, (5, 6))
    p = (3, 3)
    S = ScalarSplineSpace(p, Δ)
    S̃ = ntuple(k -> S[k], 2)

    # generate some data
    X = CartesianProduct(grevillepoints.(S̃)...)
    @evaluate C = f(X)

    # compute modal splines approximation
    rank = 2
    C̃ = approximate_patch_data(C; method=ModalSplines, spaces=S̃, rank=rank)

    # evaluate modal weighting splines at grevillepoints
    c̃ = zeros(size(C.data[1]))

    for r in 1:rank
        @evaluate U = C̃[r][1](X.data[1])
        @evaluate V = C̃[r][2](X.data[2])
        c̃ᵣ = U * V'
        c̃ += c̃ᵣ
    end

    # collect exact block data at grevillepoints
    c = C.data[1]

    # test matrix error norm
    @test norm(c - c̃) < 10e-13
end

@testset "Rank 1 modal splines approximation in three dimensions" begin
    # some analytical mapping with cross terms
    Ω = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)
    F = GeometricMapping(Ω, (x, y, z) -> x + y, (x, y, z) -> y, (x, y, z) -> z)

    # spline spaces
    Δ = Partition(Ω, (5, 6, 7))
    p = (2, 3, 4)
    S = ScalarSplineSpace(p, Δ)
    S̃ = ntuple(k -> S[k], 3)

    # generate some data
    X = CartesianProduct(grevillepoints.(S̃)...)
    @evaluate C = Gradient(F)(X)
    for k in eachindex(X)
        J = C[k]
        C[k] = inv(J)' * inv(J) * det(J)
    end

    # compute modal splines approximation
    C̃ = approximate_patch_data(C; method=ModalSplines, spaces=S̃, rank=1)

    for l in 1:2
        for k in 1:2
            # evaluate modal weighting splines at grevillepoints
            @evaluate U = C̃[1, k, l][1](X.data[1])
            @evaluate V = C̃[1, k, l][2](X.data[2])
            @evaluate W = C̃[1, k, l][3](X.data[3])

            # collect exact block data at grevillepoints
            c = C.data[k, l]

            # compute outer product of weighting splines at grevillepoints
            c̃ = W ⊗ V ⊗ U

            # test matrix error norm
            @test norm(c[:] - c̃) < 10e-13
        end
    end
end

@testset "Rank > 1 modal splines approximation in three dimensions" begin
    # some analytical mapping with cross terms
    Ω = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)
    f = ScalarFunction((x, y, z) -> x^2 + y^2 + z^2)

    # spline spaces
    Δ = Partition(Ω, (5, 6, 7))
    p = (3, 3, 3)
    S = ScalarSplineSpace(p, Δ)
    S̃ = ntuple(k -> S[k], 3)

    # generate some data
    X = CartesianProduct(grevillepoints.(S̃)...)
    @evaluate C = f(X)

    # compute modal splines approximation
    rank = 2
    C̃ = approximate_patch_data(C; method=ModalSplines, spaces=S̃, rank=rank)

    # evaluate modal weighting splines at grevillepoints
    c̃ = zeros(length(C.data[1]))

    # watch out: rank is not equal to size(C̃, 1)!
    for r in 1:rank^3
        @evaluate U = C̃[r][1](X.data[1])
        @evaluate V = C̃[r][2](X.data[2])
        @evaluate W = C̃[r][3](X.data[3])
        c̃ᵣ = W ⊗ V ⊗ U
        c̃ += c̃ᵣ
    end

    c = C.data[1]

    # test matrix error norm
    @test norm(c[:] - c̃) < 10e-13
end

@testset "CanonicalPolyadic approximation in two dimensions" begin
    # some analytical mapping with cross terms
    Ω = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)
    f = ScalarFunction((x, y) -> x^2 + y^2)

    # spline spaces
    Δ = Partition(Ω, (5, 6))
    p = (3, 3)
    S = ScalarSplineSpace(p, Δ)
    S̃ = ntuple(k -> S[k], 2)

    # generate some data
    X = CartesianProduct(grevillepoints.(S̃)...)
    @evaluate C = f(X)

    # compute approximation
    rank = 3
    C̃ = approximate_patch_data(C; method=CanonicalPolyadic, tol=10e-5, rank=rank, ntries=42)
    c̃ = zeros(length(C.data[1]))

    # compute all contributions
    for r in 1:rank
        U = C̃[r,1,1][1]
        V = C̃[r,1,1][2]
        c̃ᵣ = V ⊗ U
        c̃ += c̃ᵣ
    end

    c = C.data[1]

    # test matrix error norm
    @test norm(c[:] - c̃) < 10e-13
end

@testset "CanonicalPolyadic approximation in three dimensions" begin
    # some analytical mapping with cross terms
    Ω = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)
    f = ScalarFunction((x, y, z) -> x^2 + y^2 + z^2)

    # spline spaces
    Δ = Partition(Ω, (5, 6, 7))
    p = (3, 3, 3)
    S = ScalarSplineSpace(p, Δ)
    S̃ = ntuple(k -> S[k], 3)

    # generate some data
    X = CartesianProduct(grevillepoints.(S̃)...)
    @evaluate C = f(X)

    # compute approximation
    rank = 5
    C̃ = approximate_patch_data(C; method=CanonicalPolyadic, tol=10e-5, rank=rank, ntries=42)
    c̃ = zeros(length(C.data[1]))

    # compute all contributions
    for r in 1:rank
        U = C̃[r,1,1][1]
        V = C̃[r,1,1][2]
        W = C̃[r,1,1][3]
        c̃ᵣ = W ⊗ V ⊗ U
        c̃ += c̃ᵣ
    end

    c = C.data[1]

    # test matrix error norm
    @test norm(c[:] - c̃) < 10e-10
end

@testset "Wachspress approximation 2D" begin
    # separable
    f = ((x,y) -> 1 + x^2, (x,y) -> 2 + 2x^2)
    g = ((x,y) -> 3 + 3y^2, (x,y) -> 4 + 4y^2)

    xrange = collect(range(start=1.0, stop=3.0, length=17))
    yrange = collect(range(start=0.0, stop=1.0, length=42))
    X = CartesianProduct(xrange,yrange)
    C = IgaFormation.EvaluationSet{2,2}(IgaFormation.ResizableArray{Float64,2}, undef, size(X)...)

    for k in eachindex(X)
        x, y = X[k][1], X[k][2]
        C[k] = [ f[1](x,y) * g[1](x,y) 0.0; 0.0 f[2](x,y) * g[2](x,y) ]
    end

    D = approximate_patch_data(C; method=Wachspress, niter=2)
    for l in 1:2
        for k in 1:2
            @test norm(D[k,l][1] * D[k,l][2]' - C.data[k,l]) < 1e-12
        end
    end

    # not separable
    f = ((x,y) -> 1 + x + y, (x,y) -> 2 + 2x + y)
    g = ((x,y) -> 3 + 3y + x, (x,y) -> 4 + 4y + x)

    xrange = collect(range(start=1.0, stop=3.0, length=17))
    yrange = collect(range(start=0.0, stop=1.0, length=42))
    X = CartesianProduct(xrange,yrange)
    C = IgaFormation.EvaluationSet{2,2}(IgaFormation.ResizableArray{Float64,2}, undef, size(X)...)

    for k in eachindex(X)
        x, y = X[k][1], X[k][2]
        C[k] = [ f[1](x,y) * g[1](x,y) 0.0; 0.0 f[2](x,y) * g[2](x,y) ]
    end

    D = approximate_patch_data(C; method=Wachspress, niter=2)
    for k in 1:2
        @test (mean(C.data[k,k] ./ (D[k,k][1] * D[k,k][2]')) - 1.0) < 10e-2
    end
end

@testset "Wachspress approximation 3D" begin
    f = ((x,y,z) -> 1 + x, (x,y,z) -> 2 + 2x,  (x,y,z) -> 3 + 3x)
    g = ((x,y,z) -> 3 + 3y, (x,y,z) -> 4 + 4y, (x,y,z) -> 4 + 5y)
    h = ((x,y,z) -> 1 + 3z, (x,y,z) -> 1 + 2z, (x,y,z) -> 1 + z)

    xrange = collect(range(start=1.0, stop=3.0, length=17))
    yrange = collect(range(start=0.0, stop=1.0, length=42))
    zrange = collect(range(start=1.0, stop=2.0, length=19))
    X = CartesianProduct(xrange,yrange,zrange)
    C = IgaFormation.EvaluationSet{3,3}(IgaFormation.ResizableArray{Float64,3}, undef, size(X)...)

    for k in eachindex(X)
        x, y, z = X[k][1], X[k][2], X[k][3]
        C[k] = [ f[1](x,y,z) * g[1](x,y,z) * h[1](x,y,z) 0.0 0.0; 0.0 f[2](x,y,z) * g[2](x,y,z) * h[2](x,y,z)  0.0; 0.0 0.0 f[3](x,y,z) * g[3](x,y,z) * h[3](x,y,z) ]
    end

    D = approximate_patch_data(C; method=Wachspress, niter=2)
    for k in 1:3
        @test (mean(C.data[k,k][:] ./ (D[k,k][3] ⊗ D[k,k][2] ⊗ D[k,k][1])) - 1) < 0.2
    end
end
