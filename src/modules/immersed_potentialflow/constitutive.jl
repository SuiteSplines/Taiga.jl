"""
    PullbackBody{Dim, T <: GeometricMapping{Dim}}

Pullback of bilinear form.

# Fields:
- `mapping::T`: geometric mapping
"""
struct PullbackBody{Dim,T<:GeometricMapping{Dim}} <: AbstractMapping{Dim,Dim,Dim}
    mapping::T
    function PullbackBody(mapping::GeometricMapping{Dim}) where Dim
        T = typeof(mapping)
        new{Dim,T}(mapping)
    end
end

for Dim in 2:3
    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{$Dim,$Dim}, x::CartesianProduct{$Dim}, p::PullbackBody{$Dim})
        F = p.mapping
        @evaluate ∇F = Gradient(F)(x)
        for k in eachindex(x)  
            J = ∇F[k]
            y[k] = inv(J)' * inv(J) * det(J)
        end
        return y
    end
end

"""
    PullbackBoundary{Dim, T <: GeometricMapping{Dim}}

Pullback of linear form.

# Fields:
- `mapping::T`: geometric mapping
- `traction::Function`: traction vector
- `side::Int`: side restriction (`side <= 2Dim`)
"""
struct PullbackBoundary{Dim,T<:GeometricMapping{Dim}} <: AbstractMapping{Dim,1,1}
    mapping::T
    traction::Function
    side::Int
    function PullbackBoundary(mapping::GeometricMapping{Dim}, traction::Function, side::Int) where Dim
        T = typeof(boundary(mapping, side))
        new{Dim-1,T}(boundary(mapping, side), traction)
    end
end

for Dim = 2:3
    @eval function AbstractMappings.evalkernel!(::Val{:(=)}, y::EvaluationSet{1,1}, x::CartesianProduct{$Dim}, p::PullbackBoundary{$Dim-1})
        x, y = squeeze(x), squeeze(y)
        F, t = p.mapping, p.traction
        @evaluate X = F(x)
        @evaluate n = Normal(F)(x)
        sign = F.orientation
        for k in eachindex(x)  
            y[k] = sign * dot(t(X[k]...)', n[k])
        end
        return y
    end
end
