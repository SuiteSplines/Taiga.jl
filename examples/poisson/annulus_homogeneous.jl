using Taiga
using IgaBase, NURBS, UnivariateSplines, SpecialSpaces
using StaticArrays, LinearAlgebra, Krylov, SpecialFunctions

# define mapping
F = annulus(; inner_radius=11.064709488501170, outer_radius=17.61596604980483, β=2π)

# refine mapping (to examplify the isogeometric paradigm)
F = refine(F, method=pRefinement(3))
F = refine(F, method=hRefinement(8))

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
ūʰ = Field(S)

# define transformation from Cartesian to polar coordinates
r(x, y) = abs(x + im * y)
θ(x, y) = angle(x + im * y)

# define analytical solution
uₐ = ScalarFunction((x, y) -> besselj(4, r(x, y)) * cos(4θ(x, y)))

# define Poisson parameters
f = (x, y) -> uₐ(x, y); # bodyforce (-Δuʰ = f)
ū = (x, y) -> 0.0; # function satisfying boundary conditions
κ = (x, y) -> SMatrix{2,2,Float64}(I) # conductivity
t = (x, y) -> @SVector [0.0; 0.0] # traction

# project ū onto solution space
project!(ūʰ, onto=Field(ū), method=Interpolation)

# initialize model
@info "Model assembly..."
model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t);

# construct linear operator
L = linear_operator(model)

# construct rhs vector
b = forcing(L)

# solve unpreconditioned system
@info "Solving linear system of equations..."
solver = TaigaCG(L)
x₀, stats = linsolve!(solver, b)
@info stats

# apply particular solution
x = apply_particular_solution(L, x₀)

# apply solution to field coefficients
setcoeffs!(uʰ, S, x)

# compute L₂ error norm
L₂ = l2_error(uʰ, to=uₐ ∘ F, relative=true)[1]
@info "L₂ error norm: " L₂

# perform Bezier extraction and save results using convenience functions
@info "Exporting results..."
uₐʰ = Field(S)
project!(uₐʰ, onto=Field(uₐ ∘ F); method=Interpolation)
vtk_save_bezier("results", F; fields = Dict("uʰ" => uʰ, "uₐʰ" => uₐʰ, "ūʰ" => ūʰ))
