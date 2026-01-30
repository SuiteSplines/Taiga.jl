using Taiga, WriteVTK, NURBS

# define mapping
mapping = screw()
filepath = "control_net_screw"

# define control net context
let ctx = mapping
    points = vtk_control_net_points(ctx)
    weights = vtk_control_net_weights(ctx)
    connectivity = vtk_control_net_connectivity(ctx)
    cells = vtk_control_net_cells(connectivity)

    outfiles = vtk_grid(filepath, points..., cells) do vtk
        vtk["NurbsWeights", VTKPointData()] = weights
    end
end

# less explicitly: 
#vtk_save_control_net(filepath, mapping)