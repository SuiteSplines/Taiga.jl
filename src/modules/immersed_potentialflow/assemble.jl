"""
    assemble_body(acc::ElementAccessor{Dim}, acc_immersed::ElementAccessor{Dim}, partition::Partition{Dim}, pullback::PullbackBody{Dim}, ϕ::AlgoimCallLevelSetFunction; show_progress::Bool = true)

Assemble unconstrained stiffness matrix.

# Arguments:
- `acc`: element accessor
- `acc_immersed`: immersed element accessor
- `partition`: partition
- `pullback`: bilinear form pullback
- `ϕ`: level set function
- `show_progress`: boolean flag for formation progress bar
"""
function assemble_body(acc::ElementAccessor{Dim}, acc_immersed::ElementAccessor{Dim}, partition::Partition{Dim}, pullback::PullbackBody{Dim}, ϕ::AlgoimCallLevelSetFunction; show_progress::Bool=true) where {Dim}
    mat_cache = MatrixSumfactoryCache(acc)
    pullback_cache = FieldEvaluationCache(acc, Dim, Dim)

    m, n = length(acc.uind), prod(s -> Degree(s)+1, acc.trialfuns)
    Kₑ = zeros(n, n)

    elements = Elements(partition)
    nelem = length(elements)

    vind = 0
    nval = nelem * n^2
    rows = Vector{Int}(undef, nval)
    cols = Vector{Int}(undef, nval)
    vals = zeros(nval)

    progress = Progress(nelem, desc="Formation", color=:default, enabled=show_progress)

    for element in elements
        etest = InsideOutsideTest(ϕ, element) # todo: InsideOutsideTest on non-cartesian meshes

        if !is_outside(etest)
            Kₑ .= 0.0

            if is_inside(etest)
                ∇u = ntuple(k -> TrialFunctions(acc, element; ders=ι(k, dim=Dim)), Dim)
                ∇v = ntuple(k -> TestFunctions(acc, element; ders=ι(k, dim=Dim)), Dim)

                # optimized version for Cartesian embedding case:
                for α = 1:Dim
                    Kₑ += ∇v[α] * ∇u[α]'
                end

                #x = QuadraturePoints(acc, element)

                #Cₑ = extract_element_cache(pullback_cache, element)
                #@evaluate! Cₑ = pullback(x) # Cₑ === Id (mapping is Cartesian for now)

                #∫ = Sumfactory(mat_cache, element)
                #for β = 1:Dim
                #    for α = 1:Dim
                #        ∫(∇u[α], ∇v[β]; data=Cₑ.data[α, β], reset=false)
                #    end
                #end
                #Kₑ .= ∫.data[1]
            end

            if on_interface(etest)
                quadrule = QuadratureRule(acc_immersed, element; phase=-1)
                for k in eachindex(quadrule.w)
                    x, w = quadrule.x[k], quadrule.w[k]
                    ∇u = ntuple(d -> TrialFunctions(acc_immersed, element, k; ders=ι(d, dim=Dim)), Dim)
                    ∇v = ntuple(d -> TestFunctions(acc_immersed, element, k; ders=ι(d, dim=Dim)), Dim)

                    # optimized version for Cartesian embedding case:
                    for α = 1:Dim
                        Kₑ += w * ∇v[α] * ∇u[α]'
                    end

                    #Cₑ = SMatrix{Dim,Dim,Float64}(I) # todo: pullback evaluation at single quadrature point
                    #for β in 1:Dim
                    #    for α in 1:Dim
                    #        Kₑ += ∇v[β] * (w * Cₑ[α,β]) * ∇u[α]'
                    #    end
                    #end
                end
            end

            # collect contributions
            A = TestIndices(acc, element)
            B = TrialIndices(acc, element)
            for k in CartesianIndices(Kₑ)
                vind += 1
                rows[vind] = A[k[1]]
                cols[vind] = B[k[2]]
                vals[vind] = Kₑ[k]
            end
        end

        next!(progress)
    end

    resize!(rows, vind)
    resize!(cols, vind)
    resize!(vals, vind)
    K = sparse(rows, cols, vals, m, m)
end

"""
    assemble_boundary(acc::ElementAccessor{Dim}, acc_immersed::ElementAccessor{Dim}, partition::Partition{Dim, T}, pullbacks_boundary::NTuple{N, PullbackBoundary{M}}, ϕ::AlgoimCallLevelSetFunction)

Assemble right hand side vector.

# Arguments:
- `acc`: element accessor
- `acc_immersed`: immersed element accessor
- `partition`: partition
- `pullbacks_boundary`: tuple of `2Dim` linear form pullbacks restricted to each boundary
- `ϕ`: level set function
"""
function assemble_boundary(acc::ElementAccessor{Dim}, acc_immersed::ElementAccessor{Dim}, partition::Partition{Dim,T}, pullbacks_boundary::NTuple{N,PullbackBoundary{M}}, ϕ::AlgoimCallLevelSetFunction) where {Dim,N,M,T}
    vec_cache = VectorSumfactoryCache(acc)
    pullback_cache = FieldEvaluationCache(acc)
    traction = pullbacks_boundary[1].traction

    m, n = length(acc.uind), prod(s -> Degree(s)+1, acc.trialfuns)
    b = zeros(m)
    bₑ = zeros(n)

    # immersed boundary
    for element in Elements(partition)
        etest = InsideOutsideTest(ϕ, element)
        if !is_outside(etest)
            if on_interface(etest)
                A = TestIndices(acc, element)
                quadrule = QuadratureRule(acc_immersed, element; phase=0)
                bₑ .= 0.0

                for k in eachindex(quadrule.w)
                    v = TestFunctions(acc_immersed, element, k; ders=ι(0, dim=Dim))
                    x, w = quadrule.x[k], quadrule.w[k]
                    n = Algoim.normal(ϕ, x)
                    t = traction(x...)
                    bₑ += v * w * dot(t, n)
                end
                b[A] += bₑ
            end
        end
    end

    # embedding boundary
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
