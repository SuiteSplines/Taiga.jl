using KroneckerProducts, SortedSequences, UnivariateSplines, CartesianProducts
using NURBS, SpecialSpaces, AbstractMappings
using StaticArrays, LinearAlgebra


function cartesian_laplace()
    constraints = ScalarSplineSpaceConstraints{2}()
    left_constraint!(constraints; dim=1)
    right_constraint!(constraints; dim=1)
    left_constraint!(constraints; dim=2)
    right_constraint!(constraints; dim=2)

    Ω = Interval(0.0,1.0) ⨱ Interval(0.0,1.0)
    Δ = Partition(Ω, (7,19))
    p = (2,2)

    S = ScalarSplineSpace(p, Δ, constraints)
    C = reverse(extraction_operator(S).data)

    # mass matrices
    M₁ = Symmetric(C[1]' * UnivariateSplines.system_matrix(S[1], S[1], 1, 1) * C[1])
    M₂ = Symmetric(C[2]' * UnivariateSplines.system_matrix(S[2], S[2], 1, 1) * C[2])

    # stiffness matrices
    K₁ = Symmetric(C[1]' * UnivariateSplines.system_matrix(S[1], S[1], 2, 2) * C[1])
    K₂ = Symmetric(C[2]' * UnivariateSplines.system_matrix(S[2], S[2], 2, 2) * C[2])

    M = M₂ ⊗ K₁ + K₂ ⊗ M₁
    return Matrix.((M, M₁, M₂, K₁, K₂))
end

function poisson_annulus()
    # define mapping
    F = annulus(; inner_radius=11.064709488501170, outer_radius=17.61596604980483, β=2π)
    F = refine(F, method=pRefinement(1))
    F = refine(F, method=hRefinement(16))

    # define dimension
    Dim = dimension(F)

    # get domain mapping is defined on
    Ω = domain(F)

    # define space constraints
    C = ScalarSplineSpaceConstraints{Dim}()
    periodic_constraint!(C; c=collect(1:F.space[1].p), dim=1);
    left_constraint!(C; dim=2);
    right_constraint!(C; dim=2);

    # define scalar spline space
    S = ScalarSplineSpace(F.space)

    # define solution field
    uʰ = Field(S)

    # define field to satisfy Dirichlet boundary conditions
    ūʰ = Field(S);

    # define Poisson parameters
    f = (x, y) -> 1.0; # bodyforce (-Δuʰ = f)
    κ = (x, y) -> SMatrix{2,2,Float64}([1.0 0.95; 0.95 1.0]) # conductivity
    t = (x, y) -> @SVector [0.0; 0.0] # traction
    ūʰ[1].coeffs .= 0

    # initialize model
    model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t)
end    

function poisson_kite()
    F = rectangle()
    F = refine(F, method=pRefinement(2))
    F[1].coeffs[end,end] = F[2].coeffs[end,end] = 5 # pull out corner

    # refine mapping (to examplify the isogeometric paradigm)
    F = refine(F, method=hRefinement(16))

    # define dimension
    Dim = dimension(F)

    # get domain mapping is defined on
    Ω = domain(F)

    # define space constraints
    C = ScalarSplineSpaceConstraints{Dim}()
    left_constraint!(C; dim=1);
    right_constraint!(C; dim=1);
    left_constraint!(C; dim=2);
    right_constraint!(C; dim=2);

    # define scalar spline space
    S = ScalarSplineSpace(F.space)

    # define solution field
    uʰ = Field(S)

    # define field to satisfy Dirichlet boundary conditions
    ūʰ = Field(S)

    # define Poisson parameters
    f = (x, y) -> 1.0; # bodyforce (-Δuʰ = f)
    κ = (x, y) -> SMatrix{2,2,Float64}(I) # conductivity
    t = (x, y) -> @SVector [0.0; 0.0] # traction

    # project ū onto solution space
    ūʰ[1].coeffs .= 0

    # initialize model
    model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t);
end    