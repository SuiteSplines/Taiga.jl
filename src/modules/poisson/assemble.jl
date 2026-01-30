function assemble_body(; space, partition::Partition{Dim,T}, acc, pullback_body) where {Dim,T}
    # create cache for sum factorization
    mat_cache = MatrixSumfactoryCache(acc)
    vec_cache = VectorSumfactoryCache(acc)

    # allocate space for global mass matrix
    C = extraction_operator(space)
    m = size(C, 1)
    K = spzeros(m, m)
    b = zeros(m)

    # loop over elements
    for element in Elements(partition)

        # get global indices
        A = TestIndices(acc, element)
        B = TrialIndices(acc, element)

        # compute transformation due to material and geometry
        Cₑ, Fₑ = pullback_body(element)

        # compute trial function derivatives
        ∇u = ntuple(k -> TrialFunctions(acc, element; ders=ι(k, dim=Dim)), Dim)

        # compute test functions
        v = TestFunctions(acc, element; ders=ι(0, dim=Dim))
        ∇v = ntuple(k -> TestFunctions(acc, element; ders=ι(k, dim=Dim)), Dim)

        # compute element stiffness matrix
        ∫ = Sumfactory(mat_cache, element)
        for β = 1:Dim
            for α = 1:Dim
                ∫(∇u[α], ∇v[β]; data=Cₑ.data[α, β], reset=false)
            end
        end

        # add to global mass matrix
        K[A, B] += ∫.data[1]

        # compute element contribution of the right-hand-side
        ∫ = Sumfactory(vec_cache, element)

        # add to global rhs vector
        b[A] += ∫(v; data=Fₑ.data[1])
    end

    return K, b, C
end

function assemble_forcing!(b::Vector{T}; space, partition::Partition{Dim,T}, acc, pullback_forcing) where {Dim,T}
    # create cache for sum factorization
    vec_cache = VectorSumfactoryCache(acc)

    # compute transformation due to material and geometry
    Fₑ = pullback_forcing()

    # compute test functions
    v = TestFunctions(acc; ders=ι(0, dim=Dim))

    # compute element contribution of the right-hand-side
    ∫ = Sumfactory(vec_cache)

    # compute forcing vector
    ∫(v; data=Fₑ.data[1])
    copyto!(b, ∫.data[1])
end

function assemble_boundary!(b::Vector{T}; partition::Partition{Dim,T}, acc, mapping, pullback_boundary, traction) where {Dim,T}
    # create cache for sum factorization
    vec_cache = VectorSumfactoryCache(acc)

    # loop over boundaries
    for side in 1:2Dim

        # define pull-back of traction vector on current boundary component
        pullback_boundary!(e) = pullback_boundary_data!(acc, e,
            FieldEvaluationCache(acc), boundary(mapping, side), traction)

        # loop over boundary elements
        for element in Elements(restriction(partition, side))

            # get global indices
            A = TestIndices(acc, element)

            # compute test function derivatives at the quadrature points
            v = TestFunctions(acc, element; ders=ι(0, dim=Dim))

            # compute the pulled-back material data
            Fₜ = pullback_boundary!(element)

            # get sumfactorization object
            ∫ = Sumfactory(vec_cache, element)

            # compute contribution to force vector
            ∫(v; data=Fₜ.data[1])

            # add to global stiffness matrix
            b[A] += ∫.data[1]

        end # element loop

    end # all boundaries
end
