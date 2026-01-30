using Taiga, WriteVTK, NURBS, AbstractMappings, TensorProductBsplines
using IgaBase, CartesianProducts

# domain
interval = Interval(0.0, 1.0)
Ω = interval ⨱ interval

# some analytical mapping
F₁ = GeometricMapping(Ω, (x,y) -> x + y, (x,y) -> y)

# some analytical fields
θ = Field((x,y) -> (x+y), (x,y) -> y)
θ₁ = Field((x,y) -> (x+y))
θ₂ = Field((x,y) -> y)

# spline spaces
Δ = partition(Ω, (5, 9))
p = (3,4)
S = ScalarSplineSpace(p, Δ)
V = VectorSplineSpace(S,S)

# Nurbs projections of the analytical mapping (BezierExtraction assumes Nurbs)
F₂ = GeometricMapping(Nurbs, S.S; codimension=2)
project!(F₂, onto=F₁; method=Interpolation)

# finite dimensional fields
θʰ = Field(TensorProductBspline(V)...)
θʰ₁ = Field(TensorProductBspline(S))
θʰ₂ = Field(TensorProductBspline(S))
project!(θʰ, onto=θ; method=Interpolation)
project!(θʰ₁, onto=θ₁; method=Interpolation)
project!(θʰ₂, onto=θ₂; method=Interpolation)

# vtk file path
filepath = "bezier_extraction_fields"

# Bezier extraction of Nurbs mapping and finite dimensional θʰ
let ctx = BezierExtractionContext(F₂.space)
    degrees = vtk_bezier_degrees(ctx)
    cells = vtk_bezier_cells(ctx)
    points = vtk_extract_bezier_points(ctx, F₂)
    weights = vtk_extract_bezier_weights(ctx, F₂)
    vector_field_coeffs  = vtk_extract_bezier_points(ctx, θʰ)
    scalar_field_coeffs1 = vtk_extract_bezier_points(ctx, θʰ₁)
    scalar_field_coeffs2 = vtk_extract_bezier_points(ctx, θʰ₂)

    # write out using WriteVTK
    outfiles = vtk_grid(filepath, points..., cells; vtkversion = v"2.2") do vtk
        vtk["HigherOrderDegrees", VTKCellData()] = degrees
        vtk["RationalWeights", VTKPointData()] = weights
        vtk["θʰ", VTKPointData()] = vector_field_coeffs
        vtk["θʰ₁", VTKPointData()] = scalar_field_coeffs1
        vtk["θʰ₂", VTKPointData()] = scalar_field_coeffs2
        vtk[VTKPointData()] = "RationalWeights" => "RationalWeights"
        vtk[VTKCellData()] = "HigherOrderDegrees" => "HigherOrderDegrees"
    end
end

# less explicitly: 
#fields = Dict(
#    "θʰ" => θʰ,
#    "θʰ₁" => θʰ₁,
#    "θʰ₂" => θʰ₂
#)
#vtk_save_bezier(filepath, F₂; fields=fields)