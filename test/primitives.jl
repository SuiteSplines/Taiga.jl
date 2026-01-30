using NURBS, JLD2

@testset "Curved rectangle" begin
    F = curved_rectangle(; width=2.0, height=3.0, offset=-0.5)
    filename = "bezier_extraction_curved_rectangle"

    let ctx = BezierExtractionContext(F.space)
        degrees = vtk_bezier_degrees(ctx)
        cells = vtk_bezier_cells(ctx)
        points = vtk_extract_bezier_points(ctx, F)
        weights = vtk_extract_bezier_weights(ctx, F)

        #jldsave("assets/$filename.jld2"; degrees=degrees, cells=cells, weights=weights, points=points)
        jldopen("assets/$filename.jld2") do data
            @test all(data["degrees"] .== degrees)
            @test all(data["cells"] .== cells)
            @test all(data["weights"] .≈ weights)
            @test all(data["points"] .≈ points)
        end
    end

    vtk_save_bezier("build/$filename", F)
end

@testset "Curved cube" begin
    F = curved_cube(; width=2.0, height=3.0, depth=10.0, offset=-0.5)
    filename = "bezier_extraction_curved_cube"

    let ctx = BezierExtractionContext(F.space)
        degrees = vtk_bezier_degrees(ctx)
        cells = vtk_bezier_cells(ctx)
        points = vtk_extract_bezier_points(ctx, F)
        weights = vtk_extract_bezier_weights(ctx, F)

        #jldsave("assets/$filename.jld2"; degrees=degrees, cells=cells, weights=weights, points=points)
        jldopen("assets/$filename.jld2") do data
            @test all(data["degrees"] .== degrees)
            @test all(data["cells"] .== cells)
            @test all(data["weights"] .≈ weights)
            @test all(data["points"] .≈ points)
        end
    end

    vtk_save_bezier("build/$filename", F)
end

@testset "Pinched rectangle" begin
    F = pinched_rectangle(; width=2.0, height=3.0, c=0.001, α=π/5)
    filename = "bezier_extraction_pinched_rectangle"

    let ctx = BezierExtractionContext(F.space)
        degrees = vtk_bezier_degrees(ctx)
        cells = vtk_bezier_cells(ctx)
        points = vtk_extract_bezier_points(ctx, F)
        weights = vtk_extract_bezier_weights(ctx, F)

        #jldsave("assets/$filename.jld2"; degrees=degrees, cells=cells, weights=weights, points=points)
        jldopen("assets/$filename.jld2") do data
            @test all(data["degrees"] .== degrees)
            @test all(data["cells"] .== cells)
            @test all(data["weights"] .≈ weights)
            @test all(data["points"] .≈ points)
        end
    end

    vtk_save_bezier("build/$filename", F)
end

@testset "Pinched cube" begin
    F = pinched_cube(; width=2.0, height=3.0, depth=10.0, c=0.001, α=π/5)
    filename = "bezier_extraction_pinched_cube"

    let ctx = BezierExtractionContext(F.space)
        degrees = vtk_bezier_degrees(ctx)
        cells = vtk_bezier_cells(ctx)
        points = vtk_extract_bezier_points(ctx, F)
        weights = vtk_extract_bezier_weights(ctx, F)

        #jldsave("assets/$filename.jld2"; degrees=degrees, cells=cells, weights=weights, points=points)
        jldopen("assets/$filename.jld2") do data
            @test all(data["degrees"] .== degrees)
            @test all(data["cells"] .== cells)
            @test all(data["weights"] .≈ weights)
            @test all(data["points"] .≈ points)
        end
    end

    vtk_save_bezier("build/$filename", F)
end

@testset "Horseshoe" begin
    F = horseshoe()
    filename = "bezier_extraction_horseshoe"

    let ctx = BezierExtractionContext(F.space)
        degrees = vtk_bezier_degrees(ctx)
        cells = vtk_bezier_cells(ctx)
        points = vtk_extract_bezier_points(ctx, F)
        weights = vtk_extract_bezier_weights(ctx, F)

        #jldsave("assets/$filename.jld2"; degrees=degrees, cells=cells, weights=weights, points=points)
        jldopen("assets/$filename.jld2") do data
            @test all(data["degrees"] .== degrees)
            @test all(data["cells"] .== cells)
            @test all(data["weights"] .≈ weights)
            @test all(data["points"] .≈ points)
        end
    end

    vtk_save_bezier("build/$filename", F)
end