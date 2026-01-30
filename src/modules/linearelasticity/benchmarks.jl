"""
    BenchmarkSolution

Struct holding displacement and stress mappings for benchmark problems.

# Fields:
- `displacement::Function`: displacement vector as a function of physical coordinates
- `stress::Function`: stress tensor as a function of physical coordinates
"""
struct BenchmarkSolution
    displacement::Function
    stress::Function
end

"""
    benchmark_hole_in_plate_2d(; E, ν)

Hole in a plate benchmark in two dimensions.
"""
function benchmark_hole_in_plate_2d(; E, ν)

    # variables
    Tₓ = 10.0
    R = 1.0
    μ = E   / (2*(1+ν))
    λ = ν*E / ((1+ν)*(1-2ν))
    κ = 3-4ν  # plane strain (# κ = (3-ν) / (1+ν)   # plane stress)
    

    # analytical stress components in polar coordinates
    Σ(r,θ) = (Tₓ/2) * @SMatrix [(1 - (R/r)^2) + (1-4*(R/r)^2 + 3*(R/r)^4)*cos(2θ)       -(1 + 2(R/r)^2 - 3*(R/r)^4)*sin(2θ);
                                        -(1 + 2(R/r)^2 - 3*(R/r)^4)*sin(2θ)     (1 + (R/r)^2) - (1+3*(R/r)^4)*cos(2θ)]

    # analytical solution of the displacement
    U(r,θ) = (1/(2μ)) * @SVector [(r*Tₓ/2) * cos(2θ) * (1 + (κ+1)*(R/r)^2 - (R/r)^4) + (r*Tₓ/4)*((κ-1) + 2*(R/r)^2);
                            -(r*Tₓ/2) * sin(2θ) * (1 + (κ-1)*(R/r)^2 + (R/r)^4)]

    # transformation matrix
    Q(θ) = @SMatrix [cos(θ) -sin(θ);
                     sin(θ)  cos(θ)]

    # coordinate change (x,y) -> (r,θ)
    inverse_mapping(x,y) = sqrt.(x.^2+y.^2), atan.(y,x)

    # analytical stress components in Cartesian coordinates
    function σ(x::Real,y::Real)
        r, θ = inverse_mapping(x,y)
        return Q(θ) * Σ(r,θ) * Q(θ)'
    end

    function σ(X::AbstractMatrix, Y::AbstractMatrix)
        @assert size(X) == size(Y)
        stress = EvaluationSet{2,2}(Matrix{Float64}, undef, size(X)...)
        for j in axes(X,2)
            for i in axes(X,1)
                x, y = X[i,j], Y[i,j]
                stress[i,j] = σ(x, y)
            end
        end
        return stress
    end

    # analytical displacement components in Cartesian coordinates
    function u(x::Real,y::Real)
        r, θ = inverse_mapping(x,y)
        return Q(θ) * U(r,θ)
    end

    function u(X::AbstractMatrix, Y::AbstractMatrix)
        @assert size(X) == size(Y)
        displacement = EvaluationSet{2,1}(Matrix{Float64}, undef, size(X)...)
        for j in axes(X,2)
            for i in axes(X,1)
                x, y = X[i,j], Y[i,j]
                displacement[i,j] = u(x, y)
            end
        end
        return displacement
    end

    return BenchmarkSolution(u, σ)
end

"""
    benchmark_hole_in_plate_3d(; E, ν)

Hole in a plate benchmark in three dimensions.
"""
function benchmark_hole_in_plate_3d(; E, ν)

    b = benchmark_hole_in_plate_2d(E=E, ν=ν)

    # variables
    μ = E   / (2*(1+ν))
    λ = ν*E / ((1+ν)*(1-2ν))
    C = @SMatrix [λ+2μ  λ; λ  λ+2μ]

    # analytical stress components in Cartesian coordinates
    function u(x::Real,y::Real, z::Real)
        v = b.displacement(x,y)
        return @SVector [v[1]; v[2]; 0.0] # plain strain
    end
    
    # analytical stress components in Cartesian coordinates
    function σ(x::Real,y::Real, z::Real)
        s = b.stress(x,y)
        σ_n = @SVector [s[1,1]; s[2,2]]
        σ_zz = λ * sum(inv(C) * σ_n)
        return @SMatrix [s[1,1] s[1,2] 0.0; 
                        s[2,1] s[2,2] 0.0; 
                        0.0    0.0    σ_zz]
    end

    function σ(X::AbstractMatrix, Y::AbstractMatrix, Z::AbstractMatrix)
        @assert size(X) == size(Y) == size(Z)
        stress = EvaluationSet{3,3}(Matrix{Float64}, undef, size(X)...)
        for k in axes(X,3)
            for j in axes(X,2)
                for i in axes(X,1)
                    x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                    stress[i,j,k] = σ(x, y, z)
                end
            end
        end
        return stress
    end

    function u(X::AbstractMatrix, Y::AbstractMatrix, Z::AbstractMatrix)
        @assert size(X) == size(Y) == size(Z)
        displacement = EvaluationSet{3,1}(Matrix{Float64}, undef, size(X)...)
        for k in axes(X,3)        
            for j in axes(X,2)
                for i in axes(X,1)
                    x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                    displacement[i,j,k] = u(x, y, z)
                end
            end
        end
        return displacement
    end

    return BenchmarkSolution(u, σ)
end

"""
    benchmark_spherical_cavity(; E, ν)

Spherical cavity benchmark in three dimensions.
"""
function benchmark_spherical_cavity(; E, ν)

    # variables
    T_z = 10.0;
    R = 1.0
    μ = E / (2*(1+ν))
    A1 = T_z*ν/(1+ν);
    A2 = T_z*R^5 / (7-5*ν);
    A3 = T_z*R^3*(6-5*ν) / (2*(7-5*ν));
    B1 = -T_z / (2*(1+ν));
    B2 = -5*T_z*R^3 / (2*(7-5*ν));

    # displacement components in spherical coordinates
    U_r(r,β,θ) = (1/(2*μ)) * (-A1 * r + (3/2)*(A2/r^4) - (A3/r^2) + (3*A1*r - (9/2)*(A2/r^4) + B1*(4*ν-2)*r + (B2 * (4*ν-5) / r^2)) * cos(β)^2);
    U_β(r,β,θ) = (1/(2*μ)) * ( -3*A1*r - (3*A2/r^4) + (B1*r + (B2/r^2))*(2 - 4*ν) ) * sin(β) * cos(β);
    U_θ(r,β,θ) = 0.0;

    U(r,β,θ) = @SVector [U_r(r,β,θ);
                         U_β(r,β,θ);
                         U_θ(r,β,θ)];
    
    # stress components in spherical coordinates
    Σ_rr(r,β,θ) = T_z * cos(β)^2 + (T_z / (7-5ν)) * ((R/r)^3 * (6-5*(5-ν)*cos(β)^2 ) +  6*(R/r)^5 *(3*cos(β)^2 -1) )
    Σ_θθ(r,β,θ) = (3T_z/(2*(7-5ν))) * ( (R/r)^3*(5ν-2 + 5*(1-2ν)*cos(β)^2)  + (R/r)^5*(1-5*cos(β)^2))
    Σ_ββ(r,β,θ) = T_z * sin(β)^2 + (T_z / (2*(7-5*ν))) * ((R/r)^3 * (4-5ν + 5*(1-2ν)*cos(β)^2) + 3*(R/r)^5 * (3 - 7cos(β)^2));
    Σ_rβ(r,β,θ) = T_z * (-1 + (1/(7-5ν)) * (-5*(1+ν)*(R/r)^3 + 12*(R/r)^5)) * cos(β) * sin(β)
    
    Σ(r,β,θ) = @SMatrix [Σ_rr(r,β,θ)    Σ_rβ(r,β,θ)     0.0;
                         Σ_rβ(r,β,θ)    Σ_ββ(r,β,θ)     0.0;
                         0.0            0.0             Σ_θθ(r,β,θ)];
    
    # rotation
    Q(β, θ) = @SMatrix [sin(β)*cos(θ)   cos(β)*cos(θ)  -sin(θ); 
                        sin(β)*sin(θ)   cos(β)*sin(θ)   cos(θ);
                        cos(β)         -sin(β)          0.0];

    # coordinate change from cartesian to spherical coordinates
    function inverse_mapping(x,y,z)
        r = sqrt.(x.^2 + y.^2 + z.^2)
        β = acos.(z./r)
        θ = atan.(y,x)
        return r, β, θ
    end

    # analytical stress components in Cartesian coordinates
    function σ(x::Real, y::Real, z::Real)
        r, β, θ = inverse_mapping(x, y, z)      # coordinate change
        T = Q(β,θ)                              # transformation matrix
        return T * Σ(r,β,θ) * T'
    end

    function σ(X::AbstractMatrix, Y::AbstractMatrix, Z::AbstractMatrix)
        @assert size(X) == size(Y) == size(Z)
        stress = EvaluationSet{3,3}(Matrix{Float64}, undef, size(X)...)
        for k in axes(X,3)
            for j in axes(X,2)
                for i in axes(X,1)
                    x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                    stress[i,j,k] = σ(x, y, z)
                end
            end
        end
        return stress
    end

    # analytical displacement components in Cartesian coordinates
    function u(x::Real, y::Real, z::Real)
        r, β, θ = inverse_mapping(x, y, z)      # coordinate change
        T = Q(β,θ)                              # transformation matrix
        return T * U(r,β,θ)
    end

    function u(X::AbstractMatrix, Y::AbstractMatrix, Z::AbstractMatrix)
        @assert size(X) == size(Y) == size(Z)
        displacement = EvaluationSet{3,1}(Matrix{Float64}, undef, size(X)...)
        for k in axes(X,3)        
            for j in axes(X,2)
                for i in axes(X,1)
                    x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                    displacement[i,j,k] = u(x, y, z)
                end
            end
        end
        return displacement
    end

    return BenchmarkSolution(u, σ)
end