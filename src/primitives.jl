export screw, curved_rectangle, curved_cube, pinched_rectangle, pinched_cube, horseshoe

function screw()
    rotation(p, θ, s) = s * [cos(θ) -sin(θ); sin(θ) cos(θ)] * p
    a = 1.0
    b = sind(60) * a
    c = 0.725 * a
    S₁ = SplineSpace(Degree(1), KnotVector([0.0, a / 2, a] ./ a, [2, 1, 2]))
    S₂ = SplineSpace(Degree(1), KnotVector([0.0, a / 2, a] ./ a, [2, 1, 2]))

    square = GeometricMapping(Nurbs, S₁ ⨷ S₂; codimension=2)

    square.weights .= 1

    square[1].coeffs[1, :] .= -c / 2
    square[1].coeffs[2, :] .= 0
    square[1].coeffs[3, :] .= c / 2

    square[2].coeffs[:, 1] .= -c / 2
    square[2].coeffs[:, 2] .= 0
    square[2].coeffs[:, 3] .= c / 2

    hexagon = GeometricMapping(Nurbs, S₁ ⨷ S₂; codimension=2)

    hexagon.weights .= 1

    hexagon[1].coeffs[1, :] .= -a / 2
    hexagon[1].coeffs[2, :] .= 0
    hexagon[1].coeffs[3, :] .= a / 2

    hexagon[2].coeffs[:, 1] .= -b
    hexagon[2].coeffs[:, 2] .= 0
    hexagon[2].coeffs[:, 3] .= b

    hexagon[1].coeffs[1, 2] = -a
    hexagon[1].coeffs[3, 2] = a


    depth = 0.5a
    end_threads = 2
    nthreads = 7 + end_threads
    nthread_levels = 4 * nthreads
    quarter_thread_height = 0.175 * a
    depth_knots = [0.0, 0.05a, depth - 0.05a, depth]
    multiplicities = [3, 2, 2, 1]
    push!(depth_knots, ntuple(k -> depth_knots[end] + k * 0.1 * a, nthread_levels)...)
    push!(multiplicities, repeat([1], nthread_levels - 1)..., 3)
    S₃ = SplineSpace(Degree(2), KnotVector(depth_knots ./ depth_knots[end], multiplicities))
    head = GeometricMapping(Nurbs, hexagon.space ⨷ S₃; codimension=3)

    head.weights .= 1

    head[1].coeffs[:, :, 1] = square[1].coeffs
    head[2].coeffs[:, :, 1] = square[2].coeffs

    for k = 2:6
        head[1].coeffs[:, :, k] .= hexagon[1].coeffs
        head[2].coeffs[:, :, k] .= hexagon[2].coeffs
    end

    head[1].coeffs[:, :, 7] = square[1].coeffs
    head[2].coeffs[:, :, 7] = square[2].coeffs

    head[3].coeffs[:, :, 1] .= -0.025a
    head[3].coeffs[:, :, 2] .= 0.025a
    head[3].coeffs[:, :, 3] .= 0.05a

    head[3].coeffs[:, :, 4] .= depth / 2
    head[3].coeffs[:, :, 5] .= depth

    head[3].coeffs[:, :, 6] .= depth + 0.025a
    head[3].coeffs[:, :, 7] .= depth + 0.15a

    head_offset = 7
    for l in 1:nthread_levels-end_threads*4
        for k in eachindex(square[1].coeffs)
            x, y = square[1].coeffs[k], square[2].coeffs[k]
            x, y = rotation([x; y], l * pi / 4, 0.8)
            view(head[1].coeffs, :, :, head_offset + l)[k] = x
            view(head[2].coeffs, :, :, head_offset + l)[k] = y
            head[3].coeffs[:, :, head_offset+l] = head[3].coeffs[:, :, head_offset+l-1] .+ quarter_thread_height
        end
    end

    scaling = 0.8
    for l in nthread_levels-end_threads*4:nthread_levels
        for k in eachindex(square[1].coeffs)
            x, y = square[1].coeffs[k], square[2].coeffs[k]
            x, y = rotation([x; y], l * pi / 4, scaling)
            view(head[1].coeffs, :, :, head_offset + l)[k] = x
            view(head[2].coeffs, :, :, head_offset + l)[k] = y
            head[3].coeffs[:, :, head_offset+l] = head[3].coeffs[:, :, head_offset+l-1] .+ quarter_thread_height
        end
        scaling -= 0.8 / 8.0
    end

    return head
end

function curved_rectangle(; width::Real=1.0, height::Real=1.0, offset::Real=0.15)
    @assert width > 0 && height > 0

    S₁ = SplineSpace(Degree(2), KnotVector([0.0, 0.5, 1.0], [3, 1, 3]))
    S₂ = SplineSpace(Degree(2), KnotVector([0.0, 0.5, 1.0], [3, 1, 3]))

    surf = GeometricMapping(Nurbs, S₁ ⨷ S₂; codimension=2)

    surf.weights .= 1

    surf[1].coeffs[1, :] .= 0.0
    surf[1].coeffs[2, :] .= 1width / 6
    surf[1].coeffs[3, :] .= 5width / 6
    surf[1].coeffs[4, :] .= width

    surf[2].coeffs[:, 1] .= 0.0
    surf[2].coeffs[:, 2] .= 1height / 6
    surf[2].coeffs[:, 3] .= 5height / 6
    surf[2].coeffs[:, 4] .= height

    surf[1].coeffs[:, 2] .-= offset
    surf[1].coeffs[:, 3] .-= -offset
    surf[2].coeffs[2, :] .-= -offset
    surf[2].coeffs[3, :] .-= offset


    return surf
end

function curved_cube(; width::Real=1.0, height::Real=1.0, depth::Real=1.0, offset::Real=0.15)
    @assert width > 0 && height > 0 && depth > 0

    surf = curved_rectangle(width=width, height=height, offset=offset)
    S₃ = SplineSpace(Degree(1), KnotVector([0.0, 0.5, 1.0], [2, 1, 2]))

    volume = GeometricMapping(Nurbs, surf.space ⨷ S₃; codimension=3)

    volume.weights .= 1

    volume[1].coeffs[:, :, 1] = volume[1].coeffs[:, :, 2] = volume[1].coeffs[:, :, 3] = surf[1].coeffs
    volume[2].coeffs[:, :, 1] = volume[2].coeffs[:, :, 2] = volume[2].coeffs[:, :, 3] = surf[2].coeffs

    volume[3].coeffs[:, :, 1] .= 0.0
    volume[3].coeffs[:, :, 2] .= depth / 2
    volume[3].coeffs[:, :, 3] .= depth

    return volume
end

function pinched_cube(; width::Real=1.0, height::Real=1.0, depth::Real=1.0, c::Real=0.01, α::Real=π / 3)
    @assert width > 0 && height > 0 && depth > 0 && α ≥ 0 && α < π/2

    F = cube(; width=width, height=height, depth=depth)
    F = refine(F, method=pRefinement(1))

    R(x, u, θ) = u * dot(u, x) + cos(θ) * cross(cross(u, x), u) + sin(θ) * cross(u, x)

    F[1].coeffs .-= width / 2
    F[2].coeffs .-= height / 2
    F[3].coeffs .-= depth / 2

    F[1].coeffs[:, :, 2] .*= c
    F[2].coeffs[:, :, 2] .*= c

    for l in 1:size(F[1].coeffs, 2)
        for k in 1:size(F[1].coeffs, 1)
            x = F[1].coeffs[k, l, 1]
            y = F[2].coeffs[k, l, 1]
            z = F[3].coeffs[k, l, 1]

            X = [x, y, z]
            Y = R(X, [1, 0, 0], α)

            F[1].coeffs[k, l, 1] = Y[1]
            F[2].coeffs[k, l, 1] = Y[2]
            F[3].coeffs[k, l, 1] = Y[3]
        end
    end
    return F
end

function pinched_rectangle(; width::Real=1.0, height::Real=1.0, c::Real=0.01, α::Real=π / 3)
    @assert width > 0 && height > 0 && α ≥ 0 && α < π/2

    F = rectangle(; width=width, height=height)
    F = refine(F, method=pRefinement(1))

    R(x, θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)] * x

    F[1].coeffs .-= 0.5width
    F[2].coeffs .-= 0.5height

    F[1].coeffs[:, 2] .*= c

    for k in 1:size(F[1].coeffs, 2)
        x = F[1].coeffs[k, 1]
        y = F[2].coeffs[k, 1]

        X = [x, y]
        Y = R(X, α)

        F[1].coeffs[k, 1] = Y[1]
        F[2].coeffs[k, 1] = Y[2]
    end

    return F
end

function horseshoe()
    S₁ = SplineSpace(Degree(2), [0, 0.5, 1], [3, 1, 3])
    S₂   = SplineSpace(Degree(2), Interval(0.0, 1.0), 1)
    S₃ = SplineSpace(Degree(2), [0.0, 1/6, 1/3, 1/2, 4/6, 5/6, 1.0], [3, 1, 1, 2, 1, 1, 3])

    vol = GeometricMapping(Nurbs, S₁ ⨷ S₂ ⨷ S₃; codimension=3)

    vol[1].coeffs[:] .= [-1,-0.85355339,-0.353553388321968,0,-2.5,-2.437868,-0.97340187,0,-4,-4,-4,0,-1,-0.85355339,-0.353553388321968,0,-2.5,-2.437868,-0.97340187,0,-4,-4,-4,0,-1,-0.85355339,-0.353553388321968,0,-2.5,-2.437868,-0.97340187,0,-4,-4,-4,0,-0.70710678,-0.60355339,-0.249999998321968,0,-1.76776695,-1.72383299154504,-0.688299061941679,0,-2.82842712,-2.82842712,-2.82842712,0,0.2071,0.353553388321968,0.85355339,1.2071,-0.54289322,-0.035481605,1.4289845,1.9571068,-1.2928932,0.70710678,0.70710678,2.7071068,0.70710678,0.853553412464104,1.20710678,1.41421356,0.70710678,1.38076118676634,2.41629509515649,2.47487373,0.70710678,3.5355339,3.5355339,3.5355339,1,1.2071068124641,1.70710678,2,1,1.9526912,3.4171573,3.5,1,5,5,5,1,1.2071068124641,1.70710678,2,1,1.9526912,3.4171573,3.5,1,5,5,5,1,1.2071068124641,1.70710678,2,1,1.9526912,3.4171573,3.5,1,5,5,5]
    vol[2].coeffs[:] .= [0,0.353553388321968,0.85355339,1,0,0.95269119,2.4171573,2.5,0,4,4,4,0,0.353553388321968,0.85355339,1,0,0.95269119,2.4171573,2.5,0,4,4,4,0,0.353553388321968,0.85355339,1,0,0.95269119,2.4171573,2.5,0,4,4,4,0,0.249999998321968,0.60355339,0.70710678,0,0.673654399695268,1.70918831515649,1.76776695,0,2.82842712,2.82842712,2.82842712,-1.2071,-0.85355339,-0.353553388321968,-0.2071,-1.9571068,-1.4496952,0.014770927,0.54289322,-2.7071068,-0.70710678,-0.70710678,1.2928932,-1.41421356,-1.20710678,-0.853553412464104,-0.70710678,-2.47487373,-2.43093977154504,-1.39540586315488,-0.70710678,-3.5355339,-3.5355339,-3.5355339,-0.70710678,-2,-1.70710678,-1.2071068124641,-1,-3.5,-3.437868,-1.9734019,-1,-5,-5,-5,-1,-2,-1.70710678,-1.2071068124641,-1,-3.5,-3.437868,-1.9734019,-1,-5,-5,-5,-1,-2,-1.70710678,-1.2071068124641,-1,-3.5,-3.437868,-1.9734019,-1,-5,-5,-5,-1]
    vol[3].coeffs[:] .= [0,0,0,0,0,0,0,0,0,0,0,0,2,1.70710678,1.70710678,2,2,2,2,2,2,2,2,2,4,3.41421356,3.41421356,4,4,4,4,4,4,4,4,4,4.962333960684,4.490014734227,4.490014734227,4.962333960684,6.041803171032,6.682724756424,6.682724756424,6.041803171032,7.121343092058,10.192378548276,10.192378548276,7.121343092058,7.0178,6.349839734227,6.349839734227,7.0178,8.5444,9.4508,9.4508,8.5444,10.0711,14.4142,14.4142,10.0711,4.962333960684,4.490014734227,4.490014734227,4.962333960684,6.041803171032,6.682724756424,6.682724756424,6.041803171032,7.121343092058,10.192378548276,10.192378548276,7.121343092058,4,3.41421356,3.41421356,4,4,4,4,4,4,4,4,4,2,1.70710678,1.70710678,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0]
    vol.weights[:] .= [1,0.85355339,0.85355339,1,1,1,1,1,1,1,1,1,1,0.85355339,0.85355339,1,1,1,1,1,1,1,1,1,1,0.85355339,0.85355339,1,1,1,1,1,1,1,1,1,0.70710678,0.60355339,0.60355339,0.70710678,0.70710678,0.70710678,0.70710678,0.70710678,0.70710678,0.70710678,0.70710678,0.70710678,1,0.85355339,0.85355339,1,1,1,1,1,1,1,1,1,0.70710678,0.60355339,0.60355339,0.70710678,0.70710678,0.70710678,0.70710678,0.70710678,0.70710678,0.70710678,0.70710678,0.70710678,1,0.85355339,0.85355339,1,1,1,1,1,1,1,1,1,1,0.85355339,0.85355339,1,1,1,1,1,1,1,1,1,1,0.85355339,0.85355339,1,1,1,1,1,1,1,1,1]

    vol
end