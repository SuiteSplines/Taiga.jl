using Taiga
using NURBS, UnivariateSplines
using StaticArrays, LinearAlgebra, Krylov, SpecialFunctions

# define mapping
F = annulus(; inner_radius=11.064709488501170, outer_radius=17.61596604980483, β=2π)

# refine mapping (to examplify the isogeometric paradigm)
F = refine(F, method=pRefinement(4))
F = refine(F, method=hRefinement(32))

# define dimension
Dim = dimension(F)

# get domain mapping is defined on
Ω = domain(F)

# define space constraints
C = ScalarSplineSpaceConstraints{Dim}()
periodic_constraint!(C; c=collect(1:F.space[1].p), dim=1);
clamped_constraint!(C, :top, :bottom)

# define scalar spline space
S = ScalarSplineSpace(F.space)

# define solution field
uʰ = Field(S)

# define field to satisfy Dirichlet boundary conditions
ūʰ = Field(S)

# define transformation from Cartesian to polar coordinates
r(x, y) = abs(x + im * y)
θ(x, y) = angle(x + im * y)

# define Poisson parameters
f = (x, y) -> 1.0; # bodyforce (-Δuʰ = f)
ū = (x, y) -> 0.0; # function satisfying boundary conditions
κ = (x, y) -> SMatrix{2,2,Float64}([1.0 0.99; 0.99 1.0]) # conductivity
t = (x, y) -> @SVector [0.0; 0.0] # traction

# project ū onto solution space
project!(ūʰ, onto=Field(ū), method=Interpolation)

# initialize model
@info "Model assembly..."
model = Poisson(F, S, C, uʰ, ūʰ, κ, f, t);

# construct linear operator
L = linear_operator(model; matrixfree=true)

# construct rhs vector
b = forcing(model)

# construct linear operator approximation
p̃, ñₑ, rank = 2, 100, 2
spaces = ntuple(k -> SplineSpace(p̃, domain(F.space[k]), ñₑ), 2)
L̃ = linear_operator_approximation(model; method=ModalSplines, spaces=spaces, rank=rank)

# construct fast diagonalization preconditioner
P₁ = FastDiagonalization(model; method=Wachspress)

# construct inner cg solve preconditioner
P₂ = HyperPowerPreconditioner(L̃, P₁, 7)
@info Taiga.hyperpower_extreme_eigenvalues(P₂; n=0)

# solve system
#@info "Solving linear system of equations without preconditioning..."
#x₀, log = cg(L, b)
#@info "Solver statistics:" log

@info "Solving linear system of equations using FastDiagonalization..."
x₀, log = cg(L, b; M=P₁)
@info "Solver statistics:" log

@info "Solving linear system of equations using HyperPowerPreconditioner..."
x₀, log = cg(L, b; M=P₂)
@info "Solver statistics:" log

# apply particular solution
x = apply_particular_solution(model, x₀)

# apply solution to field coefficients
setcoeffs!(uʰ, S, x)

# perform Bezier extraction and save results using convenience functions
@info "Exporting results..."
vtk_save_bezier("results", F; fields = Dict("uʰ" => uʰ))
