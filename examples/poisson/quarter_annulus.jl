using Taiga
using IgaBase, AbstractMappings, NURBS, UnivariateSplines, SpecialSpaces
using StaticArrays, LinearAlgebra, Krylov, SpecialFunctions

# define mapping
F = annulus()

# refine mapping (to examplify the isogeometric paradigm)
F = refine(F, method=pRefinement(1))
F = refine(F, method=hRefinement(5))

# define dimension
Dim = dimension(F)

# get domain mapping is defined on
Ω = domain(F)

# define space constraints
C = ScalarSplineSpaceConstraints{Dim}()
clamped_constraint!(C, :left, :right, :bottom, :top)

# define scalar spline space
S = ScalarSplineSpace(F.space)

# define solution field
uʰ = Field(S)

# define field to satisfy Dirichlet boundary conditions
ūʰ = Field(S)

# define Poisson parameters
f = (x, y) -> 100.0; # bodyforce (-Δuʰ = f)
ū = (x, y) -> 0.0; # function satisfying boundary conditions
κ = (x, y) -> SMatrix{2,2,Float64}(I) # conductivity
t = (x, y) -> @SVector [0.0; 0.0] # traction

# project ū onto solution space
project!(ūʰ, onto=Field(ū), method=Interpolation)

# initialize model
@info "Model assembly..."
model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t);

# construct linear operator
@time L = linear_operator(model)

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

# perform Bezier extraction and save results using convenience functions
@info "Exporting results..."
vtk_save_bezier("results", F; fields = Dict("uʰ" => uʰ, "ūʰ" => ūʰ))
