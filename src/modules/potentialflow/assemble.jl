"""
    assemble_body(acc::ElementAccessor{Dim}, partition::Partition{Dim}, pullback::PullbackBody{Dim})

Assemble unconstrained stiffness matrix.

# Arguments:
- `acc`: element accessor
- `partition`: partition
- `pullback`: bilinear form pullback
"""
function assemble_body(acc::ElementAccessor{Dim}, partition::Partition{Dim}, pullback::PullbackBody{Dim}) where {Dim}
    mat_cache = MatrixSumfactoryCache(acc)
    pullback_cache = FieldEvaluationCache(acc, Dim, Dim)

    m = length(acc.uind)
    K = spzeros(m, m)

    for element in Elements(partition)
        A = TestIndices(acc, element)
        B = TrialIndices(acc, element)
        x = QuadraturePoints(acc, element)
        Cₑ = extract_element_cache(pullback_cache, element)

        v = TestFunctions(acc, element; ders=ι(0, dim=Dim))
        ∇u = ntuple(k -> TrialFunctions(acc, element; ders=ι(k, dim=Dim)), Dim)
        ∇v = ntuple(k -> TestFunctions(acc, element; ders=ι(k, dim=Dim)), Dim)

        @evaluate! Cₑ = pullback(x)

        ∫ = Sumfactory(mat_cache, element)
        for β = 1:Dim
            for α = 1:Dim
                ∫(∇u[α], ∇v[β]; data=Cₑ.data[α, β], reset=false)
            end
        end
        K[A, B] += ∫.data[1]
    end

    return K
end

"""
    assemble_boundary(acc::ElementAccessor{Dim}, partition::Partition{Dim, T}, pullbacks_boundary::NTuple{N, PullbackBoundary{M}})

Assemble right hand side vector.

# Arguments:
- `acc`: element accessor
- `partition`: partition
- `pullbacks_boundary`: tuple of `2Dim` linear form pullbacks restricted to each boundary
"""
function assemble_boundary(acc::ElementAccessor{Dim}, partition::Partition{Dim,T}, pullbacks_boundary::NTuple{N,PullbackBoundary{M}}) where {Dim,N,M,T}
    vec_cache = VectorSumfactoryCache(acc)
    pullback_cache = FieldEvaluationCache(acc)

    m = length(acc.uind)
    b = zeros(m)

    for side in 1:2Dim
        pullback = pullbacks_boundary[side]
        for element in Elements(restriction(partition, side))
            A = TestIndices(acc, element)
            x = QuadraturePoints(acc, element)
            Cₑ = extract_element_cache(pullback_cache, element)

            v = TestFunctions(acc, element; ders=ι(0, dim=Dim))

            @evaluate! Cₑ = pullback(x)

            ∫ = Sumfactory(vec_cache, element)
            ∫(v; data=Cₑ.data[1])

            b[A] += ∫.data[1]
        end
    end

    return b
end
