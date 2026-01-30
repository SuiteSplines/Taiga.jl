"""
    Model{Dim, T, M <: GeometricMapping{Dim}, V <: VectorSplineSpace{Dim, T}, W <: Field{Dim}, U <: Material{Dim}} <: Taiga.Model{Dim,T}

Linear elastic model.

# Fields:
- `F::M`: geometric mapping
- `S::V`: test and trial space
- `Δ::Partition{Dim, T}`: partition
- `uʰ::W`: solution field
- `ūʰ::W`: field compatible with boundary conditions
- `t::Function`: traction
- `material::U`: material
"""
struct Model{Dim,T,M<:GeometricMapping{Dim},V<:VectorSplineSpace{Dim,Dim,T},W<:Field{Dim},U<:Material{Dim}} <: Taiga.Model{Dim,T}
    F::M
    S::V
    Δ::Partition{Dim,T}
    uʰ::W
    ūʰ::W
    t::Function
    material::U
    function Model(F::M, S::V, C::VectorSplineSpaceConstraints{Dim}, uʰ::W, ūʰ::W, t::Function, material::U) where {Dim,T,M<:GeometricMapping{Dim},V<:VectorSplineSpace{Dim,Dim,T},W<:Field{Dim},U<:Material{Dim}}
        @assert domain(F) == domain(uʰ[1])

        args = ι(0, dim=Dim)
        @assert applicable(t, args...)

        S = VectorSplineSpace(S, C)
        Δ = Partition(S[1])
        for s in S
            @assert Δ == Partition(s)
        end

        new{Dim,T,M,V,W,U}(F, S, Δ, uʰ, ūʰ, t, material)
    end
end

"""
    LinearOperator{Dim, T} <: Taiga.LinearOperator{Dim,T}

Linear operator for linear elastic models.

# Fields:
- `C::SparseMatrixCSC{T}`: sparse extraction operator
- `K::SparseMatrixCSC{T}`: sparse stiffness matrix without constraints
- `L::SparseMatrixCSC{T}`: sparse stiffness matrix including constraints
- `b::Vector{T}`: load vector
"""
struct LinearOperator{Dim,T} <: Taiga.LinearOperator{Dim,T}
    C::SparseMatrixCSC{T}
    K::SparseMatrixCSC{T}
    L::SparseMatrixCSC{T}
    b::Vector{T}
    function LinearOperator(model::M; show_progress::Bool) where {Dim,T,M<:Model{Dim,T}}
        F, t, Δ, space, material = model.F, model.t, model.Δ, model.S, model.material
        C = blockdiag(extraction_operator.(space; sparse=true)...)

        quadrule = TensorProduct((d, u) -> PatchRule(d; npoints=ceil(Int, Degree(u) + 1), method=Legendre), Δ, space[1])
        acc = ElementAccessor(testspace=space[1], trialspace=space[1], quadrule=quadrule, incorporate_weights_in_testfuns=true)

        pullback = PullbackBilinearForm(F, material)
        pullbacks_boundary = ntuple(side -> PullbackBoundaryLinearForm(F, t, side), 2Dim)

        K = assemble_matrix(acc, Δ, pullback; show_progress=show_progress)
        b = assemble_vector(acc, Δ, pullbacks_boundary)

        L = C' * K * C
        b = C' * b

        new{Dim,T}(C, K, L, b)
    end
end

Base.size(L::LinearOperator) = (size(L.C,2), size(L.C,2))

"""
    Taiga.linear_operator(model::Model; show_progress::Bool = true)

Construct linear operator.
"""
Taiga.linear_operator(model::Model; show_progress::Bool=true) = LinearOperator(model; show_progress=show_progress)

function Taiga.LinearMaps._unsafe_mul!(b::AbstractVecOrMat, L::LinearOperator, v::AbstractVector)
    mul!(b, L.L, v)
    return b
end

"""
    SparseArrays.sparse(L::LinearOperator)

Get sparse linear operator.
"""
SparseArrays.sparse(L::LinearOperator) = L.L

"""
    Taiga.forcing!(b::AbstractVector, L::LinearOperator, model::Model)

Update forcing vector.

# Arguments:
- `b`: cache vector
- `L`: linear operator
- `model`: model
"""
function Taiga.forcing!(b::AbstractVector, L::LinearOperator{Dim}, model::Model{Dim}) where {Dim}
    b .= L.b
    b .-= L.C' * (L.K * getcoeffs(model.ūʰ)) 
end


"""
    Taiga.forcing(L::LinearOperator, model::Model)

Return forcing vector.
"""
function Taiga.forcing(L::LinearOperator, model::Model)
    b = zeros(size(L, 1))
    forcing!(b, L, model)
end

"""
    Taiga.apply_particular_solution(L::LinearOperator, model::Model, x₀::V)

Return particular solution.

# Arguments:
- `L`: linear operator
- `model`: model
- `x₀`: homogeneous solution
"""
function Taiga.apply_particular_solution(L::LinearOperator{Dim}, model::Model{Dim}, x₀::V) where {Dim,V<:AbstractVector}
    x = L.C * x₀
    x += getcoeffs(model.ūʰ)
    return x
end

