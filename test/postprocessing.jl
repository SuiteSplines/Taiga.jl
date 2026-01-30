using NURBS, AbstractMappings, IgaBase, CartesianProducts
using UnivariateSplines, TensorProductBsplines, IgaBase, SpecialSpaces
using WriteVTK, JLD2


@testset "Control net extraction utils" begin
    F = hole_in_square_plate()

    W = vtk_control_net_weights(F)
    P = vtk_control_net_points(F)

    @test W == NURBS.nurbsweights(F[1])[:]
    for k in 1:dimension(F)
        @test P[k] == UnivariateSplines.controlpoints(F[k])[:]
    end

    conn_test = [
        (1,2), (1,5), (2,3), (2,6), (3,4), (3,7), (4,8),
        (5,6), (5,9), (6,7), (6,10), (7,8), (7,11), (8,12),
        (9,10), (10,11), (11,12)
    ]
    @test conn_test == vtk_control_net_connectivity(F)
end


@testset "BezierExtractionContext (curves)" begin
    F = arc()
    ctx = BezierExtractionContext(F.space)
    @test vtk_cell_type(ctx) == VTKCellTypes.VTK_BEZIER_CURVE
    @test vtk_bezier_cell_degree(ctx) == (2,)
    @test vtk_num_cell_vertices(ctx) == 3
    @test vtk_num_cells(ctx) == 1

    F = refine(F, method=kRefinement(1,1))
    ctx = BezierExtractionContext(F.space)
    @test vtk_bezier_cell_degree(ctx) == (3,)
    @test vtk_num_cell_vertices(ctx) == 4
    @test vtk_num_cells(ctx) == 2
end


@testset "BezierExtractionContext (surfaces)" begin
    F = annulus()
    ctx = BezierExtractionContext(F.space)
    @test vtk_cell_type(ctx) == VTKCellTypes.VTK_BEZIER_QUADRILATERAL
    @test vtk_bezier_cell_degree(ctx) == (2,1)
    @test vtk_num_cell_vertices(ctx) == 3 * 2
    @test vtk_num_cells(ctx) == 1

    F = refine(F, method=kRefinement(1,1))
    ctx = BezierExtractionContext(F.space)
    @test vtk_bezier_cell_degree(ctx) == (3,2)
    @test vtk_num_cell_vertices(ctx) == 4 * 3
    @test vtk_num_cells(ctx) == 4
end


@testset "BezierExtractionContext (volumes)" begin
    F = partial_tube(; outer_radius=2.0)
    ctx = BezierExtractionContext(F.space)
    @test vtk_cell_type(ctx) == VTKCellTypes.VTK_BEZIER_HEXAHEDRON
    @test vtk_bezier_cell_degree(ctx) == (1,2,1)
    @test vtk_num_cell_vertices(ctx) == 2 * 3 * 2
    @test vtk_num_cells(ctx) == 1

    F = refine(F, method=kRefinement(1,1))
    ctx = BezierExtractionContext(F.space)
    @test vtk_bezier_cell_degree(ctx) == (2,3,2)
    @test vtk_num_cell_vertices(ctx) == 3 * 4 * 3
    @test vtk_num_cells(ctx) == 8
end


@testset "BezierExtractionContext cells" begin
    F = partial_tube(; outer_radius=2.0)
    ctx = BezierExtractionContext(F.space)
    cells = vtk_bezier_cells(ctx)
    @test first(cells) isa MeshCell
end

@testset "Bezier extraction on geometric mappings (NURBS)" begin
    F = hole_in_square_plate()
    ctx = BezierExtractionContext(F.space)
    spline_weights = NURBS.nurbsweights(F[1])
    spline_coeffs = map(NURBS.controlpoints, F)

    bezier_weights_test = ([ 1.0       1.0  1.0
                            (2+√2)/4  1.0  1.0
                            (2+√2)/4  1.0  1.0
                            (2+√2)/4  1.0  1.0
                            1.0       1.0  1.0 ])

    bezier_coeffs_test = ([  0.0       0.0   0.0
                            √2-1      0.75  4.0
                            1/√2      1.625 4.0
                            1.0       2.5   4.0
                            1.0       2.5   4.0 ])

    bezier_weights = vtk_extract_bezier_weights(ctx, F)
    bezier_coeffs = vtk_extract_bezier_points(ctx, F; vectorize=false)
    bezier_coeffs_1 = vtk_extract_bezier_points(ctx, bezier_weights_test, spline_weights, spline_coeffs[1])
    bezier_coeffs_2 = vtk_extract_bezier_points(ctx, bezier_weights_test, spline_weights, spline_coeffs[2])

    @test bezier_weights ≈ bezier_weights_test
    @test bezier_weights ≈ vtk_extract_bezier_weights(ctx, spline_weights)
    @test bezier_coeffs[1] ≈ bezier_coeffs_test
    @test bezier_coeffs[2] ≈ reverse(bezier_coeffs_test, dims=1)
    @test bezier_coeffs_1 ≈ bezier_coeffs_test
    @test bezier_coeffs_2 ≈ reverse(bezier_coeffs_test, dims=1)
end

@testset "Bezier extraction on fields (Bsplines)" begin
    Ω = Interval(0.0, 1.0) ⨱ Interval(π, 2π)
    Δ, p = Partition(Ω, (3, 4)), (2,3)
    S = ScalarSplineSpace(p, Δ) # maximum regularity
    S = ScalarSplineSpace(TensorProduct(ntuple(k -> roughen(S[k], p[k]-1), 2)...)) # minmal regularity
    V = VectorSplineSpace(S)

    θ = Field((x, y) -> (x + y), (x, y) -> y)
    θʰ = Field(V)
    project!(θʰ, onto=θ; method=Interpolation)

    ctx = BezierExtractionContext(θʰ[1].space)
    bezier_coeffs = vtk_extract_bezier_points(ctx, θʰ)

    @test bezier_coeffs[1] ≈ θʰ[1].coeffs
    @test bezier_coeffs[2] ≈ θʰ[2].coeffs
    @test vtk_extract_bezier_points(ctx, θʰ[1].coeffs) ≈ θʰ[1].coeffs
    @test vtk_extract_bezier_points(ctx, θʰ[2].coeffs) ≈ θʰ[2].coeffs
end

@testset "BezierExtractionContext cell degree" begin
    F = partial_tube(; outer_radius=2.0)
    ctx = BezierExtractionContext(F.space)
    @test vtk_bezier_cell_degree(ctx.splinespace) == map(Degree, F.space)
end

@testset "Test Bezier extraction on screw geometry" begin
    mapping = screw()
    mapping = refine(mapping, method=hRefinement(2))
    mapping = refine(mapping, method=pRefinement(1))
    filename = "bezier_extraction_screw"

    # define Bezier extraction context
    let ctx = BezierExtractionContext(mapping.space)
        degrees = vtk_bezier_degrees(ctx)
        cells = vtk_bezier_cells(ctx)
        points = vtk_extract_bezier_points(ctx, mapping)
        weights = vtk_extract_bezier_weights(ctx, mapping)

        # test writing
        outfiles = vtk_grid("build/$filename", points..., cells; vtkversion=v"2.2") do vtk
            vtk["HigherOrderDegrees", VTKCellData()] = degrees
            vtk["RationalWeights", VTKPointData()] = weights
            vtk[VTKPointData()] = "RationalWeights" => "RationalWeights"
            vtk[VTKCellData()] = "HigherOrderDegrees" => "HigherOrderDegrees"
        end

        # test data
        #jldsave("assets/$filename.jld2"; degrees=degrees, cells=cells, weights=weights, points=points)
        jldopen("assets/$filename.jld2") do data
            @test all(data["degrees"] .== degrees)
            @test all(data["cells"] .== cells)
            @test all(data["weights"] .≈ weights)
            @test all(data["points"] .≈ points)
        end
    end
end


@testset "Test Bezier extraction on hole_in_a_plate geometry" begin
    mapping = hole_in_square_plate()
    mapping = refine(mapping, method=Taiga.NURBS.pRefinement(1))
    mapping = refine(mapping, method=Taiga.NURBS.hRefinement(3))
    filename = "bezier_extraction_hole_in_a_plate"

    # define Bezier extraction context
    let ctx = BezierExtractionContext(mapping.space)
        degrees = vtk_bezier_degrees(ctx)
        cells = vtk_bezier_cells(ctx)
        points = vtk_extract_bezier_points(ctx, mapping)
        weights = vtk_extract_bezier_weights(ctx, mapping)

        # test writing
        outfiles = vtk_grid("build/$filename", points..., cells; vtkversion=v"2.2") do vtk
            vtk["HigherOrderDegrees", VTKCellData()] = degrees
            vtk["RationalWeights", VTKPointData()] = weights
            vtk[VTKPointData()] = "RationalWeights" => "RationalWeights"
            vtk[VTKCellData()] = "HigherOrderDegrees" => "HigherOrderDegrees"
        end

        # test data
        #jldsave("assets/$filename.jld2"; degrees=degrees, cells=cells, weights=weights, points=points)
        jldopen("assets/$filename.jld2") do data
            @test all(data["degrees"] .== degrees)
            @test all(data["cells"] .== cells)
            @test all(data["weights"] .≈ weights)
            @test all(data["points"] .≈ points)
        end
    end
end


@testset "Test Bezier extraction on partial_cylinder geometry" begin
    mapping = partial_cylinder()
    mapping = refine(mapping, method=Taiga.NURBS.pRefinement(1))
    filename = "bezier_extraction_partial_cylinder"

    # define Bezier extraction context
    let ctx = BezierExtractionContext(mapping.space)
        degrees = vtk_bezier_degrees(ctx)
        cells = vtk_bezier_cells(ctx)
        points = vtk_extract_bezier_points(ctx, mapping)
        weights = vtk_extract_bezier_weights(ctx, mapping)

        # test writing
        outfiles = vtk_grid("build/$filename", points..., cells; vtkversion=v"2.2") do vtk
            vtk["HigherOrderDegrees", VTKCellData()] = degrees
            vtk["RationalWeights", VTKPointData()] = weights
            vtk[VTKPointData()] = "RationalWeights" => "RationalWeights"
            vtk[VTKCellData()] = "HigherOrderDegrees" => "HigherOrderDegrees"
        end

        # test data
        #jldsave("assets/$filename.jld2"; degrees=degrees, cells=cells, weights=weights, points=points)
        jldopen("assets/$filename.jld2") do data
            @test all(data["degrees"] .== degrees)
            @test all(data["cells"] .== cells)
            @test all(data["weights"] .≈ weights)
            @test all(data["points"] .≈ points)
        end
    end
end


@testset "Test Bezier extraction on an arc geometry" begin
    mapping = arc()
    mapping = refine(mapping, method=hRefinement(3))
    filename = "bezier_extraction_arc"

    # define Bezier extraction context
    let ctx = BezierExtractionContext(mapping.space)
        degrees = vtk_bezier_degrees(ctx)
        cells = vtk_bezier_cells(ctx)
        weights = vtk_extract_bezier_weights(ctx, mapping)
        points = vtk_extract_bezier_points(ctx, mapping; bezier_weights=weights)

        # test writing
        outfiles = vtk_grid("build/$filename", points..., cells; vtkversion=v"2.2") do vtk
            vtk["HigherOrderDegrees", VTKCellData()] = degrees
            vtk["RationalWeights", VTKPointData()] = weights
            vtk[VTKPointData()] = "RationalWeights" => "RationalWeights"
            vtk[VTKCellData()] = "HigherOrderDegrees" => "HigherOrderDegrees"
        end

        # test data
        #jldsave("assets/$filename.jld2"; degrees=degrees, cells=cells, weights=weights, points=points)
        jldopen("assets/$filename.jld2") do data
            @test all(data["degrees"] .== degrees)
            @test all(data["cells"] .== cells)
            @test all(data["weights"] .≈ weights)
            @test all(data["points"] .≈ points)
        end
    end
end


@testset "Bezier extraction with fields" begin
    # domain
    Ω = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)

    # some analytical mapping
    F₁ = GeometricMapping(Ω, (x, y) -> x + y, (x, y) -> y)

    # some analytical fields
    θ = Field((x, y) -> (x + y), (x, y) -> y)
    θ₁ = Field((x, y) -> (x + y))
    θ₂ = Field((x, y) -> y)

    # spline spaces
    Δ = Partition(Ω, (5, 8))
    p = (3, 4)
    S = ScalarSplineSpace(p, Δ)
    V = VectorSplineSpace(S, S)

    # Nurbs projections of the analytical mapping (BezierExtraction assumes Nurbs)
    F₂ = GeometricMapping(Nurbs, S; codimension=2)
    project!(F₂, onto=F₁; method=Interpolation)

    # finite dimensional fields
    θʰ = Field(TensorProductBspline(V)...)
    θʰ₁ = Field(TensorProductBspline(S))
    θʰ₂ = Field(TensorProductBspline(S))
    project!(θʰ, onto=θ; method=Interpolation)
    project!(θʰ₁, onto=θ₁; method=Interpolation)
    project!(θʰ₂, onto=θ₂; method=Interpolation)

    # Bezier extraction of Nurbs mapping and finite dimensional θʰ
    let ctx = BezierExtractionContext(F₂.space)
        degrees = vtk_bezier_degrees(ctx)
        cells = vtk_bezier_cells(ctx)
        points = vtk_extract_bezier_points(ctx, F₂)
        weights = vtk_extract_bezier_weights(ctx, F₂)
        vector_field_coeffs = vtk_extract_bezier_points(ctx, θʰ)
        scalar_field_coeffs1 = vtk_extract_bezier_points(ctx, θʰ₁)
        scalar_field_coeffs2 = vtk_extract_bezier_points(ctx, θʰ₂)

        filename = "bezier_extraction_fields"

        # test writing
        outfiles = vtk_grid("build/$filename", points..., cells; vtkversion=v"2.2") do vtk
            vtk["HigherOrderDegrees", VTKCellData()] = degrees
            vtk["RationalWeights", VTKPointData()] = weights
            vtk["θʰ", VTKPointData()] = vector_field_coeffs
            vtk["θʰ₁", VTKPointData()] = scalar_field_coeffs1
            vtk["θʰ₂", VTKPointData()] = scalar_field_coeffs2
            vtk[VTKPointData()] = "RationalWeights" => "RationalWeights"
            vtk[VTKCellData()] = "HigherOrderDegrees" => "HigherOrderDegrees"
        end

        # test data
        #jldsave("assets/$filename.jld2"; degrees=degrees, cells=cells, weights=weights, points=points,
        #    vector_field_coeffs, scalar_field_coeffs1, scalar_field_coeffs2)
        jldopen("assets/$filename.jld2") do data
            @test all(data["degrees"] .== degrees)
            @test all(data["cells"] .== cells)
            @test all(data["weights"] .≈ weights)
            @test all(data["vector_field_coeffs"] .≈ vector_field_coeffs)
            @test all(data["scalar_field_coeffs1"] .≈ scalar_field_coeffs1)
            @test all(data["scalar_field_coeffs2"] .≈ scalar_field_coeffs2)
        end
    end
end


@testset "Control net on screw geometry" begin
    mapping = screw()
    filename = "control_net_screw"

    # define control net context
    let ctx = mapping
        points = vtk_control_net_points(ctx)
        weights = vtk_control_net_weights(ctx)
        connectivity = vtk_control_net_connectivity(ctx)
        cells = vtk_control_net_cells(connectivity)

        # test writing
        outfiles = vtk_grid("build/$filename", points..., cells) do vtk
            vtk["NurbsWeights", VTKPointData()] = weights
        end

        # test data
        #jldsave("assets/$filename.jld2"; cells=cells, weights=weights, points=points)
        jldopen("assets/$filename.jld2") do data
            @test all(data["cells"] .== cells)
            @test all(data["weights"] .≈ weights)
            @test all(data["points"] .≈ points)
        end
    end
end


@testset "Pretty print BezierExtractionContext" begin
    mapping = cube()
    mapping = refine(mapping, method=hRefinement(1))
    
    ctx = BezierExtractionContext(mapping.space)

    io = IOBuffer()
    Base.show(io, ctx)
    msg = String(take!(io))
    @test msg == "BezierExtraction with 8 cells of type VTK_BEZIER_HEXAHEDRON \
    á 8 vertices and univariate degrees = (1, 1, 1)."
end


@testset "Test number of Bezier cell vertices" begin
    mapping_arc = arc()
    mapping_rectangle = rectangle()
    mapping_cube = cube()

    mapping_rectangle = refine(mapping_rectangle, method=pRefinement(1))
    mapping_cube = refine(mapping_cube, method=pRefinement(2))

    ctx_arc = BezierExtractionContext(mapping_arc.space)
    ctx_rectangle = BezierExtractionContext(mapping_rectangle.space)
    ctx_cube = BezierExtractionContext(mapping_cube.space)

    @test vtk_num_cell_vertices(ctx_arc) == 3
    @test vtk_num_cell_vertices(ctx_rectangle) == 3^2
    @test vtk_num_cell_vertices(ctx_cube) == 4^3
end


@testset "Convenience save functions" begin
    # domain
    Ω = Interval(0.0, 1.0) ⨱ Interval(0.0, 1.0)

    # some analytical mapping
    F₁ = GeometricMapping(Ω, (x, y) -> x + y, (x, y) -> y)

    # some analytical fields
    θ = Field((x, y) -> (x + y), (x, y) -> y)
    θ₁ = Field((x, y) -> (x + y))
    θ₂ = Field((x, y) -> y)

    # spline spaces
    Δ = Partition(Ω, (5, 9))
    p = (3, 4)
    S = ScalarSplineSpace(p, Δ)
    V = VectorSplineSpace(S, S)

    # Nurbs projections of the analytical mapping (BezierExtraction assumes Nurbs)
    F₂ = GeometricMapping(Nurbs, S; codimension=2)
    project!(F₂, onto=F₁; method=Interpolation)

    # finite dimensional fields
    θʰ = Field(TensorProductBspline(V)...)
    θʰ₁ = Field(TensorProductBspline(S))
    θʰ₂ = Field(TensorProductBspline(S))
    project!(θʰ, onto=θ; method=Interpolation)
    project!(θʰ₁, onto=θ₁; method=Interpolation)
    project!(θʰ₂, onto=θ₂; method=Interpolation)

    # get a deep copy of coeffs
    coeffs₁ = copy(F₂[1].coeffs)
    coeffs₂ = copy(F₂[2].coeffs)

    # perform Bezier save
    fields = Dict(
        "θʰ" => θʰ,
        "θʰ₁" => θʰ₁,
        "θʰ₂" => θʰ₂
    )
    filepath = "build/vtk_save_bezierbezier_test"
    vtk_save_bezier(filepath, F₂; fields=fields)
    @test isfile("$filepath.vtu")

    # test if F₂ is still in projective space
    @test all(coeffs₁ .== F₂[1].coeffs)
    @test all(coeffs₂ .== F₂[2].coeffs)

    # perform control net save
    filepath = "build/vtk_save_control_net_test"
    vtk_save_control_net(filepath, F₂)
    @test isfile("$filepath.vtu")

    # test if F₂ is still in projective space
    @test all(coeffs₁ .== F₂[1].coeffs)
    @test all(coeffs₂ .== F₂[2].coeffs)
end