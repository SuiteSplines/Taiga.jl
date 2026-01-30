function pullback_body_data(mapping, X, κ)
    # evaluate jacobian of the mapping at all points
    @evaluate cache_X = mapping(X)
    @evaluate cache_K = Gradient(mapping)(X) # cache is reused for result
    
    # pull back transformation at each quadrature point
    for k in eachindex(X)
            
        # position at quadrature point
        x = cache_X[k]

        # Jacobian at quadrature point (i,j)
        J = cache_K[k]

        # compute and save material-geometry contribution for stiffness
        cache_K[k] = inv(J)' * κ(x...) * inv(J) * det(J)
    end

    return cache_K
end

function pullback_body_data!(cache_K, cache_F, mapping, X, κ, f)
    # evaluate jacobian of the mapping at all points
    @evaluate cache_X = mapping(X)
    @evaluate! cache_K = Gradient(mapping)(X) # cache is reused for result
    
    # pull back transformation at each quadrature point
    for k in eachindex(X)
            
        # position at quadrature point
        x = cache_X[k]

        # Jacobian at quadrature point (i,j)
        J = cache_K[k]
        
        # multiply rhs with Jacobian determinant
        cache_F[k] = f(x...) * det(J)

        # compute and save material-geometry contribution for stiffness
        cache_K[k] = inv(J)' * κ(x...) * inv(J) * det(J)
    end
end

function pullback_body_data!(acc::ElementAccessor, element::Element, cache_K::FieldEvaluationCache, cache_F::FieldEvaluationCache, mapping, κ, f)
    element_cache_K = extract_element_cache(cache_K, element)
    element_cache_F = extract_element_cache(cache_F, element)
    x = QuadraturePoints(acc, element)
    pullback_body_data!(element_cache_K, element_cache_F, mapping, x, κ, f)
    return element_cache_K, element_cache_F
end

function pullback_boundary_data!(traction_data, mapping, X, traction)
    @evaluate x = mapping(X)
    @evaluate normal = Normal(mapping)(X)
    sign = mapping.orientation

    for k in 1:length(X)
        traction_data[k] = sign * dot(traction(x[k]...)', normal[k])
    end
end

function pullback_boundary_data!(acc::ElementAccessor, element::Element, cache_traction_data, mapping, traction)
    element_cache_traction_data = extract_element_cache(cache_traction_data, element)
    x = QuadraturePoints(acc, element)
    pullback_boundary_data!(squeeze(element_cache_traction_data), mapping, squeeze(x), traction)
    return element_cache_traction_data
end

function pullback_forcing_data!(cache_F, mapping, X, κ, f)
    # evaluate jacobian of the mapping at all points
    @evaluate cache_X = mapping(X)
    @evaluate Grad = Gradient(mapping)(X) # cache is reused for result
    
    # pull back transformation at each quadrature point
    for k in eachindex(X)

        # position at quadrature point
        x = cache_X[k]

        # Jacobian at quadrature point
        J = Grad[k]
       
        # multiply rhs with Jacobian determinant
        cache_F[k] = f(x...) * det(J)
    end
end

function pullback_forcing_data!(acc::PatchAccessor, cache_F::FieldEvaluationCache, mapping, κ, f)
    patch_cache_F = extract_patch_cache(cache_F)
    x = QuadraturePoints(acc)
    pullback_forcing_data!(patch_cache_F, mapping, x, κ, f)
    return patch_cache_F
end

function pullback_body_data_matrixfree!(cache, X, Y, ∇Y, u, κ)
    # evaluate Jacobian of the solution field at all points
    @evaluate! cache = Gradient(u)(X) # the cache is reused
    
    # pull back transformation at each quadrature point
    for k in eachindex(cache)
            
        # physical coordinates
        y = Y[k]

        # Jacobian at quadrature point (i,j)
        J = ∇Y[k]

        # get the gradient of the solution
        ∇u = cache[k]

        # compute and save material-geometry contribution for stiffness
        cache[k] = inv(J)' * κ(y...) * (inv(J) * ∇u) * det(J)
    end
end

function pullback_body_data_matrixfree!(acc::PatchAccessor, cache_k::FieldEvaluationCache, Y, ∇Y, u, κ)
    patch_cache_k = extract_patch_cache(cache_k)
    X = QuadraturePoints(acc)
    pullback_body_data_matrixfree!(patch_cache_k, X, Y, ∇Y, u, κ)
    return patch_cache_k
end