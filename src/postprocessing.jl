export BezierExtractionContext, VTKHigherOrderDegrees
export vtk_point_index_from_ijk, vtk_cell_connectivity, get_bezier_basis_indices
export vtk_cell_point_indices, vtk_cell_degree, vtk_map_linear_indices
export vtk_control_net_connectivity, vtk_control_net_points, vtk_control_net_weights
export vtk_control_net_cells, vtk_num_cells
export vtk_bezier_cell, vtk_bezier_cells, vtk_bezier_points, vtk_bezier_weights
export vtk_cell_type, vtk_bezier_degrees, vtk_num_cell_vertices, vtk_bezier_cell_degree
export vtk_extract_bezier_weights, vtk_extract_bezier_points
export vtk_save_bezier, vtk_save_control_net

"""
    bezier_basis_dimension(S::TensorProduct{Dim, SplineSpace{T}})

Returns Bezier basis dimension if extracted from `S`.
"""
function bezier_basis_dimension(S::TensorProduct{Dim,SplineSpace{T}}) where {Dim,T}
    cell_degree = vtk_bezier_cell_degree(S)
    nelem = num_elements.(S)
    ntuple(k -> cell_degree[k] * nelem[k] + 1, Dim)
end

"""
    get_bezier_basis_indices(S::SplineSpace, e::Integer)

Computes univariate basis indices of a Bezier basis corresponding to Bspline space `S`.
"""
function get_bezier_basis_indices(S::SplineSpace, e::Integer)
    a = Degree(S) * (e - 1) + 1
    b = a + Degree(S)
    return a:b
end

"""
    BezierExtractionContext{Dim, T}

Bezier extraction context.

# Fields:
- `splinespace::TensorProduct{Dim, SplineSpace{T}}`: spline space
- `partition::CartesianProduct{Dim}`: partition
- `C::Array{KroneckerProduct, Dim}`: Bezier extraction operators
- `bezier_basis_dimension::NTuple{Dim}`: Bezier basis dimension
"""
struct BezierExtractionContext{Dim,T}
    splinespace::TensorProduct{Dim,SplineSpace{T}}
    partition::CartesianProduct{Dim}
    C::Array{KroneckerProduct,Dim}
    bezier_basis_dimension::NTuple{Dim}

    function BezierExtractionContext(S::TensorProduct{Dim,SplineSpace{T}}) where {Dim,T}
        partition = CartesianProduct(s -> breakpoints(s), S)
        nelem = num_elements.(S)
        bezier_basis_dim = bezier_basis_dimension(S)

        # compute Bezier cell data
        univariate_extration_operators = bezier_extraction_operator.(S)
        C = Array{KroneckerProduct,Dim}(undef, nelem...)
        for element in Elements(partition)
            # compute Kronecker product Bezier extration operator
            eind = element.index
            C[eind] = KroneckerProduct(k -> univariate_extration_operators[k][:, :, eind[k]], 1:Dim; reverse=true)
        end

        return new{Dim,T}(S, partition, C, bezier_basis_dim)
    end
end

"""
    BezierExtractionContext(S::SplineSpace{T})

Construct [`BezierExtractionContext`](@ref) from a univariate spline space.
"""
function BezierExtractionContext(S::SplineSpace{T}) where {T}
    BezierExtractionContext(TensorProduct(S)) # dirty implementation for Dim = 1
end

function Base.show(io::IO, B::BezierExtractionContext{Dim,T}) where {Dim,T}
    celltype = vtk_cell_type(B).vtk_name
    ncells = vtk_num_cells(B)
    degrees = vtk_bezier_cell_degree(B.splinespace)
    nvert = vtk_num_cell_vertices(B)
    msg = "BezierExtraction with $ncells cells of type $celltype á $nvert vertices \
    and univariate degrees = $degrees."
    print(io, msg)
end

"""
    vtk_save_bezier(filepath::String, mapping::GeometricMapping; fields::Union{Dict{String,T},Nothing}=nothing) where {T<:AbstractMapping}

Perform Bezier extraction and save a VTK file with the result.

`mapping` is a expected to be a NURBS. The optional dictionary `fields`
may contain fields defined by Bsplines.

# Arguments:
- `filepath`: output file path without extension
- `mapping`: geometric mapping
- `fields`: a dictionary of strings
"""
function vtk_save_bezier(filepath::String, mapping::GeometricMapping; fields::Union{Dict{String,T},Nothing}=nothing) where {T<:AbstractMapping}
    ctx = BezierExtractionContext(mapping.space)
    degrees = vtk_bezier_degrees(ctx)
    cells = vtk_bezier_cells(ctx)
    points = vtk_extract_bezier_points(ctx, mapping)
    weights = vtk_extract_bezier_weights(ctx, mapping)
    
    outfiles = vtk_grid(filepath, points..., cells; vtkversion=v"2.2") do vtk
        vtk["HigherOrderDegrees", VTKCellData()] = degrees
        vtk["RationalWeights", VTKPointData()] = weights
        vtk[VTKPointData()] = "RationalWeights" => "RationalWeights"
        vtk[VTKCellData()] = "HigherOrderDegrees" => "HigherOrderDegrees"
        if !isnothing(fields)
            for (label, field) in fields
                vtk[label, VTKPointData()] = vtk_extract_bezier_points(ctx, field)
            end
        end
    end
end

"""
    vtk_save_control_net(filepath::String, mapping::GeometricMapping)

Save a VTK file with the control net of a NURBS geometric mapping. The VTK
file will contain a dataset with the NURBS weights per control point.
"""
function vtk_save_control_net(filepath::String, mapping::GeometricMapping)
    points = vtk_control_net_points(mapping)
    weights = vtk_control_net_weights(mapping)
    connectivity = vtk_control_net_connectivity(mapping)
    cells = vtk_control_net_cells(connectivity)

    outfiles = vtk_grid(filepath, points..., cells) do vtk
        vtk["NurbsWeights", VTKPointData()] = weights
    end
end

"""
    vtk_cell_connectivity(B::BezierExtractionContext{Dim, T})

Returns cell connectivity for a [`BezierExtractionContext`](@ref).
"""
function vtk_cell_connectivity(B::BezierExtractionContext{Dim,T}) where {Dim,T}
    # compute Bezier cell definition
    num_bezier_elem = num_elements.(B.splinespace)
    cell_degree = vtk_bezier_cell_degree(B)
    cell_point_indices = vtk_cell_point_indices(cell_degree)
    cell_connectivity = vtk_reference_cell_connectivity(cell_point_indices)
    cell_npts = length(cell_connectivity)
    linds = LinearIndices(B.bezier_basis_dimension)

    # allocate Bezier data arrays
    connectivity = Array{NTuple{cell_npts,Int},Dim}(undef, num_bezier_elem...)

    for element in Elements(B.partition)
        eind = element.index
        bezier_cinds = ntuple(k -> get_bezier_basis_indices(B.splinespace[k], eind[k]), Dim)
        spline_cinds = ntuple(k -> IgaFormation.get_basis_indices(B.splinespace[k], eind[k]), Dim)
        connectivity[eind] = vtk_map_linear_indices(view(linds, bezier_cinds...), cell_connectivity)
    end

    return connectivity
end

"""
    vtk_extract_bezier_weights(B::BezierExtractionContext{Dim, T}, spline_weights::Array{T, Dim})

Returns Bezier weights from NURBS weights.
"""
function vtk_extract_bezier_weights(B::BezierExtractionContext{Dim,T}, spline_weights::Array{T,Dim}) where {Dim,T}
    linds = LinearIndices(B.bezier_basis_dimension)
    bezier_weights = zeros(B.bezier_basis_dimension)
    cache = zeros(vtk_bezier_cell_degree(B.splinespace) .+ 1)

    for element in Elements(B.partition)

        # collect element, spline and Bezier Cartesian indices
        eind = element.index
        bezier_cinds = ntuple(k -> get_bezier_basis_indices(B.splinespace[k], eind[k]), Dim)
        spline_cinds = ntuple(k -> IgaFormation.get_basis_indices(B.splinespace[k], eind[k]), Dim)

        # extract Bezier weights
        w = view(spline_weights, spline_cinds...)
        @kronecker! cache = B.C[eind] * w
        view(bezier_weights, bezier_cinds...) .= cache
    end

    return bezier_weights
end

"""
    vtk_extract_bezier_weights(B::BezierExtractionContext{Dim}, F::GeometricMapping{Dim, Codim})

Returns Bezier weights from a NURBS geometric mapping.
"""
function vtk_extract_bezier_weights(B::BezierExtractionContext{Dim}, F::GeometricMapping{Dim,Codim}) where {Dim,Codim}
    spline_weights = NURBS.nurbsweights(F[1])
    bezier_weights = vtk_extract_bezier_weights(B, spline_weights)
end

"""
    vtk_extract_bezier_points(B::BezierExtractionContext{Dim, T}, bezier_weights::Array{T, Dim}, spline_weights::Array{T, Dim}, spline_coeffs::Array{T, Dim})

Returns Bezier points from NURBS control points.
"""
function vtk_extract_bezier_points(B::BezierExtractionContext{Dim,T}, bezier_weights::Array{T,Dim}, spline_weights::Array{T,Dim}, spline_coeffs::Array{T,Dim}) where {Dim,T}
    linds = LinearIndices(B.bezier_basis_dimension)
    bezier_coeffs = zeros(B.bezier_basis_dimension)
    cache = zeros(vtk_bezier_cell_degree(B.splinespace) .+ 1)

    for element in Elements(B.partition)

        # collect element, spline and Bezier Cartesian indices
        eind = element.index
        bezier_cinds = ntuple(k -> get_bezier_basis_indices(B.splinespace[k], eind[k]), Dim)
        spline_cinds = ntuple(k -> IgaFormation.get_basis_indices(B.splinespace[k], eind[k]), Dim)

        # compute Bezier coefficients
        W = view(spline_weights, spline_cinds...)
        P = view(spline_coeffs, spline_cinds...)
        W̄ = view(bezier_weights, bezier_cinds...)

        # P̄ = W̄⁻¹ ⋅ C ⋅ W ⋅ P
        view(bezier_coeffs, bezier_cinds...) .= W .* P
        @kronecker! cache = B.C[eind] * view(bezier_coeffs, bezier_cinds...)
        view(bezier_coeffs, bezier_cinds...) .= (1 ./ W̄) .* cache
    end

    return bezier_coeffs
end

"""
    vtk_extract_bezier_points(B::BezierExtractionContext{Dim, T}, spline_coeffs::Array{T, Dim})

Returns Bezier points from Bspline control points (i.e. without rational weighting).
"""
function vtk_extract_bezier_points(B::BezierExtractionContext{Dim,T}, spline_coeffs::Array{T,Dim}) where {Dim,T}
    linds = LinearIndices(B.bezier_basis_dimension)
    bezier_coeffs = zeros(B.bezier_basis_dimension)
    cache = zeros(vtk_bezier_cell_degree(B.splinespace) .+ 1)

    for element in Elements(B.partition)

        # collect element, spline and Bezier Cartesian indices
        eind = element.index
        bezier_cinds = ntuple(k -> get_bezier_basis_indices(B.splinespace[k], eind[k]), Dim)
        spline_cinds = ntuple(k -> IgaFormation.get_basis_indices(B.splinespace[k], eind[k]), Dim)

        # compute Bezier coefficients
        P = view(spline_coeffs, spline_cinds...)
        @kronecker! cache = B.C[eind] * P
        view(bezier_coeffs, bezier_cinds...) .= cache
    end

    return bezier_coeffs
end


"""
    vtk_extract_bezier_points(B::BezierExtractionContext{Dim}, F::GeometricMapping{Dim, Codim}; bezier_weights = nothing, vectorize = true)

Returns Bezier weights from a NURBS geometric mapping.

If `vectorize` is `true` the result is a tuple of vector views to the points.

Precomputed Bezier weights can be passed in `bezier_weights` explicitly. If not provided, these will be
computed automatically.
"""
function vtk_extract_bezier_points(B::BezierExtractionContext{Dim}, F::GeometricMapping{Dim,Codim}; bezier_weights=nothing, vectorize=true) where {Dim,Codim}
    bezier_weights = (bezier_weights == nothing ? vtk_extract_bezier_weights(B, F) : bezier_weights)
    spline_weights = NURBS.nurbsweights(F[1])
    spline_coeffs = ntuple(k -> NURBS.controlpoints(F[k]), Codim)
    points = ntuple(k -> vtk_extract_bezier_points(B, bezier_weights, spline_weights, spline_coeffs[k]), Codim)
    points = (vectorize ? ntuple(k -> view(points[k], :), Codim) : points)
end

"""
    vtk_extract_bezier_points(B::BezierExtractionContext{Dim}, F::Field{Dim, Codim})

Returns Bezier points of a Bspline field.
"""
function vtk_extract_bezier_points(B::BezierExtractionContext{Dim}, F::Field{Dim,Codim}) where {Dim,Codim}
    C = ntuple(k -> KroneckerProduct(s -> s.C, F[k].space; reverse=true), Codim)
    shape = ntuple(k -> dimsplinespace(B.splinespace[k]), Dim)
    coeffs = ntuple(k -> zeros(shape...), Codim)

    for k in Base.OneTo(Codim)
        @kronecker! coeffs[k] = C[k] * F[k].coeffs
    end

    ntuple(k -> vtk_extract_bezier_points(B, coeffs[k]), Codim)
end

"""
    vtk_extract_bezier_points(B::BezierExtractionContext{Dim}, F::M) where {Dim,Codim,M<:AbstractMapping}

Returns Bezier points of Galerkin projection of an abstract mapping.

Consider this a fallback routine: it is used only if the mapping `F`
is not a B-spline or NURBS map. 
"""
function vtk_extract_bezier_points(B::BezierExtractionContext{Dim}, A::M) where {Dim,T1,T2,M<:AbstractMapping{Dim,T1,T2}}
    spline = TensorProductBspline(B.splinespace)
    quadrule = standard_quadrature_rule(A, spline)
    update!(spline.cache, quadrule.x)
    @evaluate y = A(quadrule.x)

    n = length(y.data)
    ntuple(k ->
    begin
        galerkin_project_imp!(GalerkinProjection, spline, quadrule, y.data[k])
        vtk_extract_bezier_points(B, spline.coeffs) 
    end, n)
end

"""
    vtk_bezier_cells(B::VTKBezierExtractionContext)

Collects all Bezier cells in `B`.
"""
function vtk_bezier_cells(B::BezierExtractionContext)
    celltype = vtk_cell_type(B)
    connectivity = vtk_cell_connectivity(B)
    map(c -> MeshCell(celltype, c), view(connectivity, :))
end

"""
    vtk_num_cells(B::BezierExtractionContext)

Returns number of Bezier cells.
"""
vtk_num_cells(B::BezierExtractionContext) = prod(num_elements.(B.splinespace))

"""
    vtk_num_cell_vertices(B::BezierExtractionContext)

Returns number of vertices.
"""
vtk_num_cell_vertices(B::BezierExtractionContext) = prod(vtk_bezier_cell_degree(B.splinespace) .+ 1)

"""
    vtk_bezier_cell_degree(S::TensorProduct{Dim, T}) where {Dim,T<:SplineSpace}

Computes a tuple of univariate degrees for a Bezier cell extracted from a tensor product spline space.
"""
function vtk_bezier_cell_degree(S::TensorProduct{Dim,T}) where {Dim,T<:SplineSpace}
    ntuple(k -> Degree(S[k]), Dim)
end

"""
    vtk_bezier_cell_degree(S::BezierExtractionContext{Dim, T}) where {Dim,T<:SplineSpace}

Computes univariate degrees for a Bezier cell in a `BezierExtractionContext`.
"""
function vtk_bezier_cell_degree(B::BezierExtractionContext{Dim,T}) where {Dim,T}
    vtk_bezier_cell_degree(B.splinespace)
end

"""
    vtk_cell_type(::VTKBezierExtraction{Dim}) where {Dim}

Returns a Bezier cell type based on `Dim`.
"""
vtk_cell_type(::BezierExtractionContext{1}) = VTKCellTypes.VTK_BEZIER_CURVE
vtk_cell_type(::BezierExtractionContext{2}) = VTKCellTypes.VTK_BEZIER_QUADRILATERAL
vtk_cell_type(::BezierExtractionContext{3}) = VTKCellTypes.VTK_BEZIER_HEXAHEDRON

"""
    vtk_reference_cell_connectivity(point_indices::Array{Int, Dim}) where {Dim}

Computes cell connectivity of a reference VTK Bezier cell with points
indexed using Cartesian indices.
"""
function vtk_reference_cell_connectivity(point_indices::Array{Int,Dim}) where {Dim}
    ntuple(k -> findfirst(isequal(k), view(point_indices, :)), length(point_indices))
end

"""
    vtk_control_net_connectivity(mapping::GeometricMapping{Dim})

Generates a list of connectivity tuples for all edges of a control net based on the size
of the control points grid in mapping.
"""
function vtk_control_net_connectivity(mapping::GeometricMapping{Dim}) where {Dim}
    gridsize = size(mapping[1].coeffs)
    linds = LinearIndices(gridsize)
    cinds = CartesianIndices(gridsize)
    numedges = sum(ntuple(l -> prod(ntuple(k -> (k == l) ? (gridsize[k] - 1) : gridsize[k], Dim)), Dim))
    connectivity = Vector{Tuple{Int,Int}}(undef, numedges)
    c = 1
    for I in cinds
        for d in Base.OneTo(Dim)
            J = CartesianIndex(ntuple(k -> (k == d ? I[k] + 1 : I[k]), Dim))
            if J[d] <= gridsize[d]
                connectivity[c] = (linds[I], linds[J])
                c += 1
            end
        end
    end
    return connectivity
end

"""
    vtk_control_net_weights(mapping::GeometricMapping)

Returns vectorized NURBS weights (for `WriteVTK` purposes).
"""
vtk_control_net_weights(mapping::GeometricMapping) = view(NURBS.nurbsweights(mapping[1]), :)

"""
    vtk_control_net_points(mapping::GeometricMapping{Dim, Codim})


Returns vectorized NURBS coefficients (for `WriteVTK` purposes).
"""
vtk_control_net_points(mapping::GeometricMapping{Dim,Codim}) where {Dim,Codim} = ntuple(k -> view(UnivariateSplines.controlpoints(mapping[k]), :), Codim)

"""
    vtk_control_net_cells(connectivity::Vector{Tuple{Int, Int}})

Returns a vector of `VTKCellTypes.VTK_LINE` cells given a vector of point connectivities.
"""
vtk_control_net_cells(connectivity::Vector{Tuple{Int,Int}}) = map(c -> MeshCell(VTKCellTypes.VTK_LINE, c), view(connectivity, :))

"""
    vtk_map_linear_indices(lind::AbstractArray, connectivity::NTuple{N, Int}) where {N}

Maps linear indices of a Bezier cell index by Cartesian indices based on the
connectivity of a reference VTK Bezier cell.
"""
function vtk_map_linear_indices(lind::AbstractArray, connectivity::NTuple{N,Int}) where {N}
    @assert length(lind) == N
    ntuple(k -> lind[connectivity[k]], N)
end

"""
    vtk_cell_point_indices(order::NTuple{Dim, Int}) where {Dim}

Computes point indices of a reference VTK Bezier cell with points indexed
using Cartesian indices.
"""
function vtk_cell_point_indices(order::NTuple{Dim,Int}) where {Dim}
    inds = CartesianIndices(ntuple(k -> 1:order[k]+1, Dim))
    point_indices = Array{Int,Dim}(undef, order .+ 1...)
    for ind in inds
        point_indices[ind] = vtk_point_index_from_ijk(ind, order)
    end
    return point_indices
end

"""
    vtk_point_index_from_ijk(inds::CartesianIndex{3}, order::NTuple{3, Int})

Computes VTK point index based on Cartesian index of a cell point.

Adapted from VTK's `vtkHigherOrderHexahedron::PointIndexFromIJK()`.
"""
function vtk_point_index_from_ijk(inds::CartesianIndex{3}, order::NTuple{3,Int})
    # adapted from https://gitlab.kitware.com/vtk/vtk/-/blob/master/Common/DataModel/vtkHigherOrderHexahedron.cxx#L602
    # changes: indexing from 1, ternary operators

    i = inds[1] - 1
    j = inds[2] - 1
    k = inds[3] - 1

    ibdy = (i == 0 || i == order[0+1])
    jbdy = (j == 0 || j == order[1+1])
    kbdy = (k == 0 || k == order[2+1])

    # How many boundaries do we lie on at once?
    nbdy = (ibdy > 0 ? 1 : 0) + (jbdy > 0 ? 1 : 0) + (kbdy > 0 ? 1 : 0)

    if (nbdy == 3) # Vertex DOF
        # ijk is a corner node. Return the proper index (somewhere in [0,7]):
        return (i > 0 ? (j > 0 ? 2 : 1) : (j > 0 ? 3 : 0)) + (k > 0 ? 4 : 0) + 1
    end

    offset = 8
    if (nbdy == 2) # Edge DOF
        if (!ibdy)
            # On i axis
            return (i - 1) + (j > 0 ? order[0+1] + order[1+1] - 2 : 0) + (k > 0 ? 2 * (order[0+1] + order[1+1] - 2) : 0) + offset + 1
        end
        if (!jbdy)
            # On j axis
            return (j - 1) + (i > 0 ? order[0+1] - 1 : 2 * (order[0+1] - 1) + order[1+1] - 1) + (k > 0 ? 2 * (order[0+1] + order[1+1] - 2) : 0) + offset + 1
        end
        # !kbdy, On k axis

        offset += 4 * (order[0+1] - 1) + 4 * (order[1+1] - 1)

        return (k - 1) + (order[2+1] - 1) * (i > 0 ? (j > 0 ? 2 : 1) : (j > 0 ? 3 : 0)) + offset + 1 # vtk version 2.2
        #return (k - 1) + (order[2+1] - 1) * (i > 0 ? (j > 0 ? 3 : 1) : (j > 0 ? 2 : 0)) + offset + 1 # vtk version 1.0
    end

    offset += 4 * (order[0+1] + order[1+1] + order[2+1] - 3)
    if (nbdy == 1) # Face DOF
        if (ibdy) # On i-normal face
            return (j - 1) + ((order[1+1] - 1) * (k - 1)) + (i > 0 ? (order[1+1] - 1) * (order[2+1] - 1) : 0) + offset + 1
        end
        offset += 2 * (order[1+1] - 1) * (order[2+1] - 1)
        if (jbdy) # On j-normal face
            return (i - 1) + ((order[0+1] - 1) * (k - 1)) + (j > 0 ? (order[2+1] - 1) * (order[0+1] - 1) : 0) + offset + 1
        end
        offset += 2 * (order[2+1] - 1) * (order[0+1] - 1)
        # kbdy, On k-normal face
        return (i - 1) + ((order[0+1] - 1) * (j - 1)) + (k > 0 ? (order[0+1] - 1) * (order[1+1] - 1) : 0) + offset + 1
    end

    # nbdy == 0: Body DOF
    offset += 2 * ((order[1+1] - 1) * (order[2+1] - 1) + (order[2+1] - 1) * (order[0+1] - 1) + (order[0+1] - 1) * (order[1+1] - 1))
    return offset + (i - 1) + (order[0+1] - 1) * ((j - 1) + (order[1+1] - 1) * ((k - 1))) + 1
end

"""
    vtk_point_index_from_ijk(inds::CartesianIndex{2}, order::NTuple{2, Int})

Computes VTK point index based on Cartesian index of a cell point.

Adapted from VTK's `vtkHigherOrderHexahedron::PointIndexFromIJK()`.
"""
function vtk_point_index_from_ijk(inds::CartesianIndex{2}, order::NTuple{2,Int})
    i = inds[1] - 1
    j = inds[2] - 1

    ibdy = (i == 0 || i == order[0+1])
    jbdy = (j == 0 || j == order[1+1])

    # How many boundaries do we lie on at once?
    nbdy = (ibdy > 0 ? 1 : 0) + (jbdy > 0 ? 1 : 0)
    if (nbdy == 2) # Vertex DOF
        # ijk is a corner node. Return the proper index (somewhere in [0,7]):
        return (i > 0 ? (j > 0 ? 2 : 1) : (j > 0 ? 3 : 0)) + 1
    end

    offset = 4
    if (nbdy == 1) # Edge DOF
        if (!ibdy)
            # On i axis
            return (i - 1) + (j > 0 ? order[0+1] - 1 + order[1+1] - 1 : 0) + offset + 1
        end
        if (!jbdy)
            # On j axis
            return (j - 1) + (i > 0 ? order[0+1] - 1 : 2 * (order[0+1] - 1) + order[1+1] - 1) + offset + 1
        end
    end

    offset += 2 * (order[0+1] - 1 + order[1+1] - 1)
    # nbdy == 0: Face DOF
    return offset + (i - 1) + (order[0+1] - 1) * ((j - 1)) + 1
end

"""
    vtk_point_index_from_ijk(inds::CartesianIndex{1}, order::NTuple{2, Int})

Computes VTK point index based on Cartesian index of a cell point.
"""
function vtk_point_index_from_ijk(ind::CartesianIndex{1}, order::Tuple{Int})
    ind, order = ind[1], order[1]
    if ind == 1
        return 1
    elseif ind == (order + 1)
        return 2
    else
        return (ind + 1)
    end
end

"""
    VTKHigherOrderDegrees{Dim, T}

An immutable sparse container for `WriteVTK` higher order degrees array.

This container acts like an array of size `(3, ncells)` with repeated columns
and supports indexing and views.

# Fields:
- `degrees::NTuple{Dim, T}`: cell degrees
- `ncells::Int`: number of Bezier cells
"""
struct VTKHigherOrderDegrees <: AbstractArray{Int,2}
    degrees::NTuple{3,Int}
    ncells::Int
    function VTKHigherOrderDegrees(degrees::NTuple{3,Int}, ncells::Int)
        new(degrees, ncells)
    end
end

function VTKHigherOrderDegrees(B::BezierExtractionContext{3})
    vtk_degrees = vtk_bezier_cell_degree(B.splinespace)
    ncells = prod(vtk_num_cells(B))
    VTKHigherOrderDegrees(vtk_degrees, ncells)
end

function VTKHigherOrderDegrees(B::BezierExtractionContext{2})
    bezier_degrees = vtk_bezier_cell_degree(B.splinespace)
    vtk_degrees = (bezier_degrees..., 0)
    ncells = prod(vtk_num_cells(B))
    VTKHigherOrderDegrees(vtk_degrees, ncells)
end

function VTKHigherOrderDegrees(B::BezierExtractionContext{1})
    bezier_degrees = vtk_bezier_cell_degree(B.splinespace)
    vtk_degrees = (bezier_degrees..., 0, 0)
    ncells = prod(vtk_num_cells(B))
    VTKHigherOrderDegrees(vtk_degrees, ncells)
end

Base.size(M::VTKHigherOrderDegrees) = (3, M.ncells)

function higher_order_degrees_array_index(M::VTKHigherOrderDegrees, i::Int, j::Int)
    (i % 3 == 0) ? 3 : i % 3
end

function Base.getindex(M::VTKHigherOrderDegrees, i::Int, j::Int)
    ind = higher_order_degrees_array_index(M, i, j)
    M.degrees[ind]
end

"""
    vtk_bezier_degrees(B::VTKBezierExtraction)

Returns an immutable sparse array container of type `VTKHigherOrderDegrees` with cell degrees.
"""
vtk_bezier_degrees(B::BezierExtractionContext) = VTKHigherOrderDegrees(B)