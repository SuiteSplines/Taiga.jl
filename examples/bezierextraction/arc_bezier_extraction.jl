using Taiga, WriteVTK, NURBS

# define mapping
mapping = arc()
mapping = refine(mapping, method=hRefinement(10))

# define Bezier extraction context
let ctx = BezierExtractionContext(mapping.space)
    degrees = vtk_bezier_degrees(ctx)
    cells = vtk_bezier_cells(ctx)
    weights = vtk_extract_bezier_weights(ctx, mapping)
    points = vtk_extract_bezier_points(ctx, mapping; bezier_weights = weights)

    outfiles = vtk_grid("bezier_extraction_arc", points..., cells; vtkversion = v"2.2") do vtk
        vtk["HigherOrderDegrees", VTKCellData()] = degrees
        vtk["RationalWeights", VTKPointData()] = weights
        vtk[VTKPointData()] = "RationalWeights" => "RationalWeights"
        vtk[VTKCellData()] = "HigherOrderDegrees" => "HigherOrderDegrees"
    end
end
