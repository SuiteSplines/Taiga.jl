"""
    PullbackBilinearForm{Dim, T1 <: GeometricMapping{Dim}, T2 <: Material{Dim}}

Pullback of the bilinear form.

# Fields:
- `mapping::T1`: geometric mapping
- `material::T2`: material
- `B::BMatrix{Dim}`: B-matrices.
"""
struct PullbackBilinearForm{Dim,T1<:GeometricMapping{Dim},T2<:Material{Dim}} <: AbstractMapping{Dim,Dim,Dim}
    mapping::T1
    material::T2
    B::BMatrix{Dim}
    function PullbackBilinearForm(mapping::GeometricMapping{Dim}, material::Material{Dim}) where Dim
        T1 = typeof(mapping)
        T2 = typeof(material)
        new{Dim,T1,T2}(mapping, material, BMatrix{Dim}())
    end
end

for Dim in 2:3
    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::Matrix{E}, x::CartesianProduct{$Dim}, p::PullbackBilinearForm{$Dim,S,T}) where {S,T,E<:EvaluationSet{$Dim}}
        F, B, material = p.mapping, p.B, p.material

        @evaluate ∇F = Gradient(F)(x)
        @evaluate C = material(x)

        for t = 1:$Dim
            for s = 1:$Dim
                for k in eachindex(x)
                    J = ∇F[k]
                    y[s,t][k] = (inv(J)' * B[s]' * C[k] * B[t] * inv(J)) * det(J)
                end
            end
        end

        return y
    end
end

# PullbackBilinearForm is special: it evaluates into an array of EvaluationSets
@inline function AbstractMappings.get_evaluation_cache(p::PullbackBilinearForm{Dim}, x::S) where {Dim,T,S<:CartesianProduct{Dim,NTuple{Dim,T}}}
    [EvaluationSet{Dim,Dim}(ResizableArray{T,Dim}, undef, size(x)...) for k = 1:Dim, l = 1:Dim]
end
@inline function AbstractMappings.evaluate_imp!(op, y::Matrix{E}, x, p::PullbackBilinearForm{Dim}) where {Dim,E<:EvaluationSet{Dim}}
    AbstractMappings.evalkernel!(op, y, x, p)
    return y
end

"""
    PullbackBoundaryLinearForm{Dim, S}

Pullback of the linear form on the boundary. The parameter `1 ≤ S ≤ 2Dim`
denotes the coressponding boundary face.

# Fields:
- `mapping::GeometricMapping{Dim}`: geometric mapping
- `traction::Function`: traction tensor as a function of physical coordinates
"""
struct PullbackBoundaryLinearForm{Dim,S} <: AbstractMapping{Dim,Dim,1}
    mapping::GeometricMapping{Dim}
    traction::Function
    function PullbackBoundaryLinearForm(mapping::GeometricMapping{Dim}, traction::Function, side::Int) where Dim
        new{Dim,side}(mapping, traction)
    end
end

for Dim = 2:3
    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{$Dim,1}, x::CartesianProduct{$Dim}, p::PullbackBoundaryLinearForm{$Dim,S}) where {S}
        ∂F = boundary(p.mapping,S)
        t = p.traction
        x, y = squeeze(x), squeeze(y)

        @evaluate X = ∂F(x)
        @evaluate n = Normal(∂F)(x)
        sign = ∂F.orientation

        for k in eachindex(x)
            y[k] = sign * t(X[k]...) * n[k]
        end

        return y
    end
end
