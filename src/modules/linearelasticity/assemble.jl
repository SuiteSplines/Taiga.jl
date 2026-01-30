"""
    assemble_matrix(acc::ElementAccessor{Dim}, partition::Partition{Dim}, pullback::PullbackBilinearForm{Dim}; show_progress::Bool = true)

Assemble stiffness matrix. 

# Arguments:
- `acc`: element accessor
- `partition`: partition
- `pullback`: pullback of bilinear form
- `show_progress`: boolean flag for formation progress bar
"""
function assemble_matrix(acc::ElementAccessor{Dim}, partition::Partition{Dim}, pullback::PullbackBilinearForm{Dim}; show_progress::Bool=true) where Dim
    mat_cache = MatrixSumfactoryCache(acc)
    pullback_cache = [FieldEvaluationCache(acc, Dim, Dim) for i in 1:Dim, j in 1:Dim]

    m, n = length(acc.uind), prod(s -> Degree(s)+1, acc.trialfuns)
    Kₑ = [zeros(n, n) for i in 1:Dim, j in 1:Dim]

    elements = Elements(partition)
    nelem = length(elements)

    nval = nelem * n^2
    vind = [0 for i in 1:Dim, j in 1:Dim]
    rows = [Vector{Int}(undef, nval) for i in 1:Dim, j in 1:Dim]
    cols = [Vector{Int}(undef, nval) for i in 1:Dim, j in 1:Dim]
    vals = [zeros(nval) for i in 1:Dim, j in 1:Dim]

    progress = Progress(nelem, desc="Formation", color=:default, enabled=show_progress)

    for element in Elements(partition)
        A = TestIndices(acc, element)
        B = TrialIndices(acc, element)

        ∇u = ntuple(k -> TrialFunctions(acc, element; ders=ι(k, dim=Dim)), Dim)
        ∇v = ntuple(k -> TestFunctions(acc, element; ders=ι(k, dim=Dim)), Dim)
        x = QuadraturePoints(acc, element)

        Cₑ = [extract_element_cache(pullback_cache[i,j], element) for i in 1:Dim, j in 1:Dim]
        @evaluate! Cₑ = pullback(x)

        for j in 1:Dim
            for i in 1:Dim
                Kₑ[i,j] .= 0.0

                ∫ = Sumfactory(mat_cache, element)
                for beta in 1:Dim
                    for alpha in 1:Dim    
                        ∫(∇u[beta], ∇v[alpha]; data=Cₑ[i,j].data[alpha,beta], reset=false)
                    end
                end

                Kₑ[i,j] += ∫.data[1]
            end
        end

        for j in 1:Dim
            for i in 1:Dim
                for k in CartesianIndices(Kₑ[i,j])
                    vind[i,j] += 1
                    rows[i,j][vind[i,j]] = A[k[1]]
                    cols[i,j][vind[i,j]] = B[k[2]]
                    vals[i,j][vind[i,j]] = Kₑ[i,j][k]
                end
            end
        end

        next!(progress)
    end
    finish!(progress)

    # assemble sparse blocks into one sparse matrix: note the behavior of sparse_hvcat!
    blocks = [sparse(rows[i,j], cols[i,j], vals[i,j], m, m) for j in 1:Dim, i in 1:Dim]
    K = sparse_hvcat(ntuple(d -> Dim, Dim), blocks...)
end

"""
    assemble_vector(acc::ElementAccessor{Dim}, partition::Partition{Dim}, pullbacks::NTuple{N, PullbackBoundaryLinearForm})

Assemble forcing vector.

# Arguments:
- `acc`: element accessor
- `partition`: partition
- `pullbacks`: tuple of pullbacks of the linear form on each of the boundaries
"""
function assemble_vector(acc::ElementAccessor{Dim}, partition::Partition{Dim}, pullbacks::NTuple{N,PullbackBoundaryLinearForm}) where {Dim,N}
    vec_cache = VectorSumfactoryCache(acc)
    pullback_cache = FieldEvaluationCache(acc, Dim, 1)

    m = length(acc.uind)
    b = [zeros(m) for i = 1:Dim]

    for side in 1:2Dim
        pullback = pullbacks[side]
        for element in Elements(restriction(partition, side))
            A = TestIndices(acc, element)
            v = TestFunctions(acc, element; ders=ι(0, dim=Dim))
            x = QuadraturePoints(acc, element)

            Fₜ = extract_element_cache(pullback_cache, element)
            @evaluate! Fₜ = pullback(x)

            for i in 1:Dim
                ∫ = Sumfactory(vec_cache, element)
                ∫(v; data=Fₜ.data[i], reset=false)
                b[i][A] += ∫.data[1]
            end
        end
    end

    return vcat(b...)
end